import Foundation
import Metal

public struct DenoiseWindow: Hashable, Sendable { public var key:WindowKey; public var region:Region2D; public var phase:DiffusionPhaseParameters; public init(key:WindowKey, region:Region2D, phase:DiffusionPhaseParameters){self.key=key;self.region=region;self.phase=phase} }
public struct DenoiseBatch { public var commandBuffer:MTLCommandBuffer; public var input:MetalTensor; public var conditioning:[String:MetalTensor]; public var windows:[DenoiseWindow]; public var phase:DiffusionPhaseParameters; public init(commandBuffer:MTLCommandBuffer,input:MetalTensor,conditioning:[String:MetalTensor],windows:[DenoiseWindow],phase:DiffusionPhaseParameters){self.commandBuffer=commandBuffer;self.input=input;self.conditioning=conditioning;self.windows=windows;self.phase=phase} }

public protocol MetalDenoiser: AnyObject { var name:String {get}; var inputChannels:Int {get}; var outputChannels:Int {get}; var tileHeight:Int {get}; var tileWidth:Int {get}; var legalBatchSizes:[Int] {get}; func encode(_ batch:DenoiseBatch, context:MetalContext) throws -> MetalTensor }

public final class IdentityDenoiser: MetalDenoiser {
    public let name:String; public let inputChannels:Int; public let outputChannels:Int; public let tileHeight:Int; public let tileWidth:Int; public let legalBatchSizes:[Int]
    public init(name:String, channels:Int, tileHeight:Int, tileWidth:Int, legalBatchSizes:[Int]=[1,2,4,8,16]) {
        self.name=name; self.inputChannels=channels; self.outputChannels=channels; self.tileHeight=tileHeight; self.tileWidth=tileWidth; self.legalBatchSizes=legalBatchSizes
    }
    public init(name:String, inputChannels:Int, outputChannels:Int, tileHeight:Int, tileWidth:Int, legalBatchSizes:[Int]=[1,2,4,8,16]) {
        self.name=name; self.inputChannels=inputChannels; self.outputChannels=outputChannels; self.tileHeight=tileHeight; self.tileWidth=tileWidth; self.legalBatchSizes=legalBatchSizes
    }
    public func encode(_ batch:DenoiseBatch, context:MetalContext)throws->MetalTensor {
        if batch.input.shape.c == outputChannels { return batch.input }
        let out = try context.allocate(shape: TensorShape(n: batch.input.shape.n, c: outputChannels, h: batch.input.shape.h, w: batch.input.shape.w), scalarType: .float32, storageMode: .storageModePrivate, label: "\(name).identity")
        try KernelEncoder(context: context).copyRegion(commandBuffer: batch.commandBuffer, source: batch.input, destination: out, sourceLocal: Region2D(y: 0, x: 0, height: batch.input.shape.h, width: batch.input.shape.w), destinationLocal: Region2D(y: 0, x: 0, height: batch.input.shape.h, width: batch.input.shape.w), channels: outputChannels)
        return out
    }
}

public final class CompositeDenoiser: MetalDenoiser { public let name:String; public let stages:[MetalDenoiser]; public var inputChannels:Int{stages.first!.inputChannels}; public var outputChannels:Int{stages.last!.outputChannels}; public var tileHeight:Int{stages.first!.tileHeight}; public var tileWidth:Int{stages.first!.tileWidth}; public var legalBatchSizes:[Int]{stages.reduce(stages.first?.legalBatchSizes ?? [1]){Array(Set($0).intersection($1.legalBatchSizes)).sorted()}}; public init(name:String, stages:[MetalDenoiser]) throws { guard !stages.isEmpty else { throw TerrainDiffusionError.invalidModelArchive("Composite requires stages") }; self.name=name; self.stages=stages }; public func encode(_ batch:DenoiseBatch, context:MetalContext)throws->MetalTensor{ var cur=batch.input; for s in stages { var b=batch; b.input=cur; cur=try s.encode(b, context:context) }; return cur } }
