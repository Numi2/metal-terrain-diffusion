import Foundation
import Metal

public final class MetalContext {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else { throw TerrainDiffusionError.metalUnavailable }
        guard let q = device.makeCommandQueue() else { throw TerrainDiffusionError.commandBufferFailed("makeCommandQueue") }
        self.device = device
        self.commandQueue = q
        do {
            self.library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw TerrainDiffusionError.shaderLoadFailed(String(describing: error))
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); if let p = pipelines[name] { lock.unlock(); return p }; lock.unlock()
        guard let fn = library.makeFunction(name: name) else { throw TerrainDiffusionError.shaderLoadFailed("Missing kernel \(name)") }
        let pso: MTLComputePipelineState
        do { pso = try device.makeComputePipelineState(function: fn) }
        catch { throw TerrainDiffusionError.pipelineCreationFailed("\(name): \(error)") }
        lock.lock(); pipelines[name] = pso; lock.unlock()
        return pso
    }

    public func makeCommandBuffer(label: String? = nil) throws -> MTLCommandBuffer {
        guard let cb = commandQueue.makeCommandBuffer() else { throw TerrainDiffusionError.commandBufferFailed("makeCommandBuffer") }
        cb.label = label
        return cb
    }

    public func runAndWait(_ cb: MTLCommandBuffer) throws {
        cb.commit(); cb.waitUntilCompleted()
        if let error = cb.error { throw TerrainDiffusionError.commandBufferFailed(error.localizedDescription) }
    }

    public func allocate(shape: TensorShape, scalarType: ScalarType = .float32, storageMode: MTLResourceOptions = .storageModePrivate, label: String? = nil) throws -> MetalTensor {
        let bytes = shape.byteCount(scalarType)
        guard let buffer = device.makeBuffer(length: bytes, options: storageMode) else { throw TerrainDiffusionError.allocationFailed(bytes) }
        buffer.label = label
        return MetalTensor(buffer: buffer, shape: shape, scalarType: scalarType)
    }

    public func upload(_ values: [Float], shape: TensorShape, label: String? = nil, privateStorage: Bool = true) throws -> MetalTensor {
        guard values.count == shape.elementCount else { throw TerrainDiffusionError.invalidShape("upload count \(values.count) != shape \(shape.elementCount)") }
        let bytes = values.count * MemoryLayout<Float>.stride
        guard let staging = device.makeBuffer(bytes: values, length: bytes, options: .storageModeShared) else { throw TerrainDiffusionError.allocationFailed(bytes) }
        if !privateStorage { return MetalTensor(buffer: staging, shape: shape, scalarType: .float32) }
        let out = try allocate(shape: shape, scalarType: .float32, storageMode: .storageModePrivate, label: label)
        let cb = try makeCommandBuffer(label: "upload.\(label ?? "tensor")")
        guard let blit = cb.makeBlitCommandEncoder() else { throw TerrainDiffusionError.commandBufferFailed("makeBlitCommandEncoder") }
        blit.copy(from: staging, sourceOffset: 0, to: out.buffer, destinationOffset: 0, size: bytes)
        blit.endEncoding(); try runAndWait(cb)
        return out
    }

    public func download(_ tensor: MetalTensor) throws -> [Float] {
        guard tensor.scalarType == .float32 else { throw TerrainDiffusionError.unsupportedScalarType("download currently supports float32") }
        guard let staging = device.makeBuffer(length: tensor.byteCount, options: .storageModeShared) else { throw TerrainDiffusionError.allocationFailed(tensor.byteCount) }
        let cb = try makeCommandBuffer(label: "download")
        guard let blit = cb.makeBlitCommandEncoder() else { throw TerrainDiffusionError.commandBufferFailed("makeBlitCommandEncoder") }
        blit.copy(from: tensor.buffer, sourceOffset: tensor.byteOffset, to: staging, destinationOffset: 0, size: tensor.byteCount)
        blit.endEncoding(); try runAndWait(cb)
        return Array(UnsafeBufferPointer(start: staging.contents().bindMemory(to: Float.self, capacity: tensor.elementCount), count: tensor.elementCount))
    }
}

public extension MTLComputeCommandEncoder {
    func dispatch1D(pipeline: MTLComputePipelineState, count: Int) {
        let w = min(max(pipeline.threadExecutionWidth, 1), 256)
        dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
    func dispatch2D(pipeline: MTLComputePipelineState, width: Int, height: Int) {
        let tw = min(max(pipeline.threadExecutionWidth, 1), 16)
        let th = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / max(tw, 1)))
        dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }
    func dispatch3D(pipeline: MTLComputePipelineState, width: Int, height: Int, depth: Int) {
        let tw = min(max(pipeline.threadExecutionWidth, 1), 16)
        let th = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / max(tw, 1)))
        dispatchThreads(MTLSize(width: width, height: height, depth: depth), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }
}
