import Foundation
import Metal

public struct UInt4Uniforms { public var a,b,c,d: UInt32; public init(_ a: UInt32=0,_ b: UInt32=0,_ c: UInt32=0,_ d: UInt32=0){self.a=a;self.b=b;self.c=c;self.d=d} }
public struct Int8Uniforms { public var a,b,c,d,e,f,g,h: Int32; public init(_ a:Int=0,_ b:Int=0,_ c:Int=0,_ d:Int=0,_ e:Int=0,_ f:Int=0,_ g:Int=0,_ h:Int=0){self.a=Int32(a);self.b=Int32(b);self.c=Int32(c);self.d=Int32(d);self.e=Int32(e);self.f=Int32(f);self.g=Int32(g);self.h=Int32(h)} }

public final class KernelEncoder {
    public let context: MetalContext
    public init(context: MetalContext) { self.context = context }

    private func encoder(_ cb: MTLCommandBuffer, _ kernel: String) throws -> (MTLComputeCommandEncoder, MTLComputePipelineState) {
        let pso = try context.pipeline(kernel)
        guard let enc = cb.makeComputeCommandEncoder() else { throw TerrainDiffusionError.commandBufferFailed("compute encoder") }
        enc.setComputePipelineState(pso)
        return (enc,pso)
    }

    public func fill(commandBuffer: MTLCommandBuffer, tensor: MetalTensor, value: Float) throws {
        let (enc,pso) = try encoder(commandBuffer, "td_fill_float")
        enc.setBuffer(tensor.buffer, offset: tensor.byteOffset, index: 0); var v=value; var n=UInt32(tensor.elementCount)
        enc.setBytes(&v, length: 4, index: 1); enc.setBytes(&n, length: 4, index: 2)
        enc.dispatch1D(pipeline: pso, count: tensor.elementCount); enc.endEncoding()
    }

    public func gaussian(commandBuffer: MTLCommandBuffer, out: MetalTensor, seed: UInt64, originY: Int, originX: Int, tileH: Int = 256, tileW: Int = 256, sigma: Float = 1) throws {
        let (enc,pso) = try encoder(commandBuffer, "td_gaussian_nchw")
        enc.setBuffer(out.buffer, offset: out.byteOffset, index: 0)
        var dims=UInt4Uniforms(UInt32(out.shape.c),UInt32(out.shape.h),UInt32(out.shape.w),UInt32(out.shape.n))
        var origin=Int8Uniforms(originY,originX,tileH,tileW,0,0,0,0)
        var lo=UInt32(seed & 0xffffffff), hi=UInt32((seed >> 32) & 0xffffffff), sig=sigma
        enc.setBytes(&dims,length:MemoryLayout<UInt4Uniforms>.stride,index:1); enc.setBytes(&origin,length:MemoryLayout<Int8Uniforms>.stride,index:2)
        enc.setBytes(&lo,length:4,index:3); enc.setBytes(&hi,length:4,index:4); enc.setBytes(&sig,length:4,index:5)
        enc.dispatch3D(pipeline:pso,width:out.shape.w,height:out.shape.h,depth:out.shape.c*out.shape.n); enc.endEncoding()
    }

    public func packWeighted(commandBuffer: MTLCommandBuffer, input: MetalTensor, outputPacked: MetalTensor, epsilon: Float = 1e-3) throws {
        guard outputPacked.shape.c == input.shape.c + 1, outputPacked.shape.h == input.shape.h, outputPacked.shape.w == input.shape.w, outputPacked.shape.n == input.shape.n else { throw TerrainDiffusionError.invalidShape("packWeighted output must be C+1") }
        let (enc,pso)=try encoder(commandBuffer,"td_pack_linear_weight"); enc.setBuffer(input.buffer,offset:input.byteOffset,index:0); enc.setBuffer(outputPacked.buffer,offset:outputPacked.byteOffset,index:1)
        var dims=UInt4Uniforms(UInt32(input.shape.c),UInt32(input.shape.h),UInt32(input.shape.w),UInt32(input.shape.n)); var eps=epsilon
        enc.setBytes(&dims,length:MemoryLayout<UInt4Uniforms>.stride,index:2); enc.setBytes(&eps,length:4,index:3)
        enc.dispatch3D(pipeline:pso,width:input.shape.w,height:input.shape.h,depth:outputPacked.shape.c*outputPacked.shape.n); enc.endEncoding()
    }

    public func normalizePacked(commandBuffer: MTLCommandBuffer, inputPacked: MetalTensor, output: MetalTensor, epsilon: Float = 1e-6) throws {
        guard inputPacked.shape.c == output.shape.c + 1, inputPacked.shape.h == output.shape.h, inputPacked.shape.w == output.shape.w, inputPacked.shape.n == output.shape.n else { throw TerrainDiffusionError.invalidShape("normalizePacked input must be C+1") }
        let (enc,pso)=try encoder(commandBuffer,"td_normalize_packed"); enc.setBuffer(inputPacked.buffer,offset:inputPacked.byteOffset,index:0); enc.setBuffer(output.buffer,offset:output.byteOffset,index:1)
        var dims=UInt4Uniforms(UInt32(output.shape.c),UInt32(output.shape.h),UInt32(output.shape.w),UInt32(output.shape.n)); var eps=epsilon
        enc.setBytes(&dims,length:MemoryLayout<UInt4Uniforms>.stride,index:2); enc.setBytes(&eps,length:4,index:3)
        enc.dispatch3D(pipeline:pso,width:output.shape.w,height:output.shape.h,depth:output.shape.c*output.shape.n); enc.endEncoding()
    }

    public func accumulateWindow(commandBuffer: MTLCommandBuffer, sourcePacked: MetalTensor, destinationPacked: MetalTensor, sourceLocal: Region2D, destinationLocal: Region2D) throws {
        let (enc,pso)=try encoder(commandBuffer,"td_accumulate_window"); enc.setBuffer(sourcePacked.buffer,offset:sourcePacked.byteOffset,index:0); enc.setBuffer(destinationPacked.buffer,offset:destinationPacked.byteOffset,index:1)
        var dims=Int8Uniforms(sourcePacked.shape.c,sourcePacked.shape.h,sourcePacked.shape.w,destinationPacked.shape.h,destinationPacked.shape.w,sourceLocal.y,sourceLocal.x,destinationLocal.y)
        var extra=Int8Uniforms(destinationLocal.x,sourceLocal.height,sourceLocal.width,0,0,0,0,0)
        enc.setBytes(&dims,length:MemoryLayout<Int8Uniforms>.stride,index:2); enc.setBytes(&extra,length:MemoryLayout<Int8Uniforms>.stride,index:3)
        enc.dispatch3D(pipeline:pso,width:sourceLocal.width,height:sourceLocal.height,depth:sourcePacked.shape.c); enc.endEncoding()
    }

    public func copyRegion(commandBuffer: MTLCommandBuffer, source: MetalTensor, destination: MetalTensor, sourceLocal: Region2D, destinationLocal: Region2D, channels: Int? = nil) throws {
        let c = channels ?? min(source.shape.c,destination.shape.c)
        let (enc,pso)=try encoder(commandBuffer,"td_copy_region"); enc.setBuffer(source.buffer,offset:source.byteOffset,index:0); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:1)
        var dims=Int8Uniforms(c,source.shape.h,source.shape.w,destination.shape.h,destination.shape.w,sourceLocal.y,sourceLocal.x,destinationLocal.y)
        var extra=Int8Uniforms(destinationLocal.x,sourceLocal.height,sourceLocal.width,0,0,0,0,0)
        enc.setBytes(&dims,length:MemoryLayout<Int8Uniforms>.stride,index:2); enc.setBytes(&extra,length:MemoryLayout<Int8Uniforms>.stride,index:3)
        enc.dispatch3D(pipeline:pso,width:sourceLocal.width,height:sourceLocal.height,depth:c); enc.endEncoding()
    }

    public func copyTileToBatch(commandBuffer: MTLCommandBuffer, source: MetalTensor, destinationBatch: MetalTensor, batchIndex: Int) throws {
        let (enc,pso)=try encoder(commandBuffer,"td_copy_tile_to_batch"); enc.setBuffer(source.buffer,offset:source.byteOffset,index:0); enc.setBuffer(destinationBatch.buffer,offset:destinationBatch.byteOffset,index:1)
        var dims=Int8Uniforms(source.shape.c,source.shape.h,source.shape.w,destinationBatch.shape.n,0,0,0,0); var b=UInt32(batchIndex)
        enc.setBytes(&dims,length:MemoryLayout<Int8Uniforms>.stride,index:2); enc.setBytes(&b,length:4,index:3)
        enc.dispatch3D(pipeline:pso,width:source.shape.w,height:source.shape.h,depth:source.shape.c); enc.endEncoding()
    }

    public func extractBatchTile(commandBuffer: MTLCommandBuffer, sourceBatch: MetalTensor, destination: MetalTensor, batchIndex: Int) throws {
        let (enc,pso)=try encoder(commandBuffer,"td_extract_batch_tile"); enc.setBuffer(sourceBatch.buffer,offset:sourceBatch.byteOffset,index:0); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:1)
        var dims=Int8Uniforms(destination.shape.c,destination.shape.h,destination.shape.w,sourceBatch.shape.n,0,0,0,0); var b=UInt32(batchIndex)
        enc.setBytes(&dims,length:MemoryLayout<Int8Uniforms>.stride,index:2); enc.setBytes(&b,length:4,index:3)
        enc.dispatch3D(pipeline:pso,width:destination.shape.w,height:destination.shape.h,depth:destination.shape.c); enc.endEncoding()
    }

    public func unaryScale(commandBuffer: MTLCommandBuffer, source: MetalTensor, destination: MetalTensor, scale: Float, bias: Float = 0) throws {
        try source.assertSameShape(as: destination); let (enc,pso)=try encoder(commandBuffer,"td_unary_scale")
        enc.setBuffer(source.buffer,offset:source.byteOffset,index:0); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:1)
        var s=scale,b=bias,n=UInt32(source.elementCount); enc.setBytes(&s,length:4,index:2); enc.setBytes(&b,length:4,index:3); enc.setBytes(&n,length:4,index:4)
        enc.dispatch1D(pipeline:pso,count:source.elementCount); enc.endEncoding()
    }

    public func linearMix(commandBuffer: MTLCommandBuffer, a: MetalTensor, b: MetalTensor, destination: MetalTensor, weightA: Float, weightB: Float) throws {
        try a.assertSameShape(as:b); try a.assertSameShape(as:destination); let (enc,pso)=try encoder(commandBuffer,"td_linear_mix")
        enc.setBuffer(a.buffer,offset:a.byteOffset,index:0); enc.setBuffer(b.buffer,offset:b.byteOffset,index:1); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:2)
        var wa=weightA,wb=weightB,n=UInt32(destination.elementCount); enc.setBytes(&wa,length:4,index:3); enc.setBytes(&wb,length:4,index:4); enc.setBytes(&n,length:4,index:5)
        enc.dispatch1D(pipeline:pso,count:destination.elementCount); enc.endEncoding()
    }

    public func catChannels(commandBuffer: MTLCommandBuffer, a: MetalTensor, b: MetalTensor, destination: MetalTensor) throws {
        let (enc,pso)=try encoder(commandBuffer,"td_cat_channels"); enc.setBuffer(a.buffer,offset:a.byteOffset,index:0); enc.setBuffer(b.buffer,offset:b.byteOffset,index:1); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:2)
        var dims=Int8Uniforms(a.shape.c,b.shape.c,destination.shape.h,destination.shape.w,destination.shape.n,0,0,0); enc.setBytes(&dims,length:MemoryLayout<Int8Uniforms>.stride,index:3)
        enc.dispatch3D(pipeline:pso,width:destination.shape.w,height:destination.shape.h,depth:destination.shape.c*destination.shape.n); enc.endEncoding()
    }

    public func conv2D(commandBuffer: MTLCommandBuffer, source: MetalTensor, weight: MetalTensor, destination: MetalTensor, kernelH: Int, kernelW: Int, groups: Int = 1, strideY: Int = 1, strideX: Int = 1, padY: Int? = nil, padX: Int? = nil) throws {
        let (enc,pso)=try encoder(commandBuffer,"td_conv2d_nchw"); enc.setBuffer(source.buffer,offset:source.byteOffset,index:0); enc.setBuffer(weight.buffer,offset:weight.byteOffset,index:1); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:2)
        var d0=Int8Uniforms(source.shape.n,source.shape.c,source.shape.h,source.shape.w,destination.shape.c,kernelH,kernelW,groups)
        var d1=Int8Uniforms(padY ?? kernelH/2,padX ?? kernelW/2,destination.shape.h,destination.shape.w,strideY,strideX,0,0)
        enc.setBytes(&d0,length:MemoryLayout<Int8Uniforms>.stride,index:3); enc.setBytes(&d1,length:MemoryLayout<Int8Uniforms>.stride,index:4)
        enc.dispatch3D(pipeline:pso,width:destination.shape.w,height:destination.shape.h,depth:destination.shape.c*destination.shape.n); enc.endEncoding()
    }

    public func mpSilu(commandBuffer: MTLCommandBuffer, source: MetalTensor, destination: MetalTensor) throws { try unaryOp(commandBuffer,"td_mp_silu",source,destination) }
    public func inverseSignedSqrt(commandBuffer: MTLCommandBuffer, source: MetalTensor, destination: MetalTensor) throws { try unaryOp(commandBuffer,"td_inverse_signed_sqrt",source,destination) }
    public func signedSqrt(commandBuffer: MTLCommandBuffer, source: MetalTensor, destination: MetalTensor) throws { try unaryOp(commandBuffer,"td_signed_sqrt",source,destination) }

    private func unaryOp(_ cb: MTLCommandBuffer, _ kernel: String, _ source: MetalTensor, _ destination: MetalTensor) throws {
        try source.assertSameShape(as:destination); let (enc,pso)=try encoder(cb,kernel); enc.setBuffer(source.buffer,offset:source.byteOffset,index:0); enc.setBuffer(destination.buffer,offset:destination.byteOffset,index:1); var n=UInt32(source.elementCount); enc.setBytes(&n,length:4,index:2); enc.dispatch1D(pipeline:pso,count:source.elementCount); enc.endEncoding()
    }
}
