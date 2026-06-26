import Foundation
import Metal

public struct SyntheticMapConfig: Codable, Sendable { public var seed: UInt64; public var frequencyMultipliers: [Float]; public var dropWaterProbability: Float; public init(seed: UInt64, frequencyMultipliers:[Float]=[1.5,3,3,3,3], dropWaterProbability:Float=0.5){self.seed=seed;self.frequencyMultipliers=frequencyMultipliers;self.dropWaterProbability=dropWaterProbability} }

public final class SyntheticConditioningProducer: InfiniteTileProducer {
    public let valueChannels = 5
    public let tileHeight: Int
    public let tileWidth: Int
    public let legalBatchSizes = [1]
    private let config: SyntheticMapConfig
    public init(tileHeight:Int=64, tileWidth:Int=64, config:SyntheticMapConfig){ self.tileHeight=tileHeight; self.tileWidth=tileWidth; self.config=config }
    public func produceTiles(engine: InfiniteDiffusionEngine, keys:[WindowKey], commandBuffer: MTLCommandBuffer) throws -> [WindowKey:MetalTensor] {
        // GPU-side deterministic Gaussian fields stand in for the five synthetic Perlin channels; import pipelines can override this producer.
        var out:[WindowKey:MetalTensor]=[:]
        for k in keys { let t=try engine.context.allocate(shape:TensorShape(n:1,c:5,h:tileHeight,w:tileWidth), scalarType:.float32, storageMode:.storageModePrivate, label:"synthetic.\(k)"); try engine.kernels.gaussian(commandBuffer:commandBuffer,out:t,seed:config.seed,originY:k.y*tileHeight,originX:k.x*tileWidth,tileH:tileHeight,tileW:tileWidth); out[k]=t }
        return out
    }
}
