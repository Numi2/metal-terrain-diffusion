import Foundation
import Metal

public struct TerrainDiffusionConfiguration: Codable, Sendable {
    public var seed: UInt64 = 0x1234_5678_9abc_def0
    public var T: Int = 2
    public var latentCompression: Int = 8
    public var decoderTileSize: Int = 512
    public var decoderTileStride: Int = 384
    public var latentBatchSizes: [Int] = [1,2,4,8,16]
    public var residualMean: Float = 0
    public var residualStd: Float = 1.1678
    public var lowFrequencyMean: Float = -31.4
    public var lowFrequencyStd: Float = 38.6
    public init() {}
}

public final class DecoderTileProducer: InfiniteTileProducer {
    public let valueChannels = 1
    public let tileHeight: Int
    public let tileWidth: Int
    public let legalBatchSizes = [1]
    private let denoiser: MetalDenoiser
    private let latentTensorID: String
    private let latentWindow: WindowSpec
    private let seed: UInt64
    private let latentCompression: Int
    private let phase: DiffusionPhaseParameters
    public init(denoiser: MetalDenoiser, latentTensorID: String, latentWindow: WindowSpec, tileSize: Int, seed: UInt64, latentCompression: Int, phase: DiffusionPhaseParameters) { self.denoiser=denoiser; self.latentTensorID=latentTensorID; self.latentWindow=latentWindow; self.tileHeight=tileSize; self.tileWidth=tileSize; self.seed=seed; self.latentCompression=latentCompression; self.phase=phase }
    public func produceTiles(engine: InfiniteDiffusionEngine, keys: [WindowKey], commandBuffer: MTLCommandBuffer) throws -> [WindowKey: MetalTensor] {
        var out:[WindowKey:MetalTensor]=[:]
        for k in keys {
            let region = latentWindow.region(for: WindowKey(tensorID: latentTensorID, y: k.y, x: k.x))
            let lat = try engine.queryNormalized(tensorID: latentTensorID, region: region, commandBuffer: commandBuffer)
            let up = try TerrainTransforms(context: engine.context).nearestUpsample(lat, height: tileHeight, width: tileWidth, commandBuffer: commandBuffer)
            let noise = try engine.context.allocate(shape: TensorShape(n:1,c:1,h:tileHeight,w:tileWidth), scalarType:.float32, storageMode:.storageModePrivate, label:"decoder.noise")
            try engine.kernels.gaussian(commandBuffer:commandBuffer,out:noise,seed:seed,originY:k.y*tileHeight,originX:k.x*tileWidth,tileH:tileHeight,tileW:tileWidth,sigma:phase.sigmaData)
            let input = try engine.context.allocate(shape: TensorShape(n:1,c:up.shape.c+1,h:tileHeight,w:tileWidth), scalarType:.float32, storageMode:.storageModePrivate, label:"decoder.input")
            try engine.kernels.catChannels(commandBuffer:commandBuffer, a:noise, b:up, destination:input)
            let result = try denoiser.encode(DenoiseBatch(commandBuffer:commandBuffer,input:input,conditioning:[:],windows:[DenoiseWindow(key:k,region:Region2D(y:k.y*tileHeight,x:k.x*tileWidth,height:tileHeight,width:tileWidth),phase:phase)],phase:phase), context:engine.context)
            out[k] = result
        }
        return out
    }
}

public final class TerrainDiffusionPipeline {
    public let context: MetalContext
    public let engine: InfiniteDiffusionEngine
    public let configuration: TerrainDiffusionConfiguration
    public let transforms: TerrainTransforms

    public init(context: MetalContext, configuration: TerrainDiffusionConfiguration = TerrainDiffusionConfiguration(), tileStore: TileStore? = nil) {
        self.context = context
        self.configuration = configuration
        self.engine = InfiniteDiffusionEngine(context: context, tileStore: tileStore)
        self.transforms = TerrainTransforms(context: context)
    }

    public static func fromArchives(context: MetalContext, root: URL, configuration: TerrainDiffusionConfiguration = TerrainDiffusionConfiguration(), persistentCache: URL? = nil) throws -> TerrainDiffusionPipeline {
        let store: TileStore? = try persistentCache.map { try PersistentTileStore(context: context, root: $0) }
        let p = TerrainDiffusionPipeline(context: context, configuration: configuration, tileStore: store)
        let coarse = try MetalGraphDenoiser(context: context, archiveURL: root.appendingPathComponent("coarse_model.metalgraph"))
        let base = try MetalGraphDenoiser(context: context, archiveURL: root.appendingPathComponent("base_model.metalgraph"))
        let decoder = try MetalGraphDenoiser(context: context, archiveURL: root.appendingPathComponent("decoder_model.metalgraph"))
        try p.buildStages(coarse: coarse, base: base, decoder: decoder)
        return p
    }

    public func buildDebugIdentityStages() throws {
        try buildStages(
            coarse: IdentityDenoiser(name:"coarse.identity", channels:6, tileHeight:64, tileWidth:64),
            base: IdentityDenoiser(name:"base.identity", channels:5, tileHeight:64, tileWidth:64),
            decoder: IdentityDenoiser(name:"decoder.identity", inputChannels:6, outputChannels:1, tileHeight:configuration.decoderTileSize, tileWidth:configuration.decoderTileSize)
        )
    }

    public func buildStages(coarse: MetalDenoiser, base: MetalDenoiser, decoder: MetalDenoiser) throws {
        let conditioningWindow = try WindowSpec(channels:6, tileHeight:64, tileWidth:64, strideY:48, strideX:48)
        let coarseWindow = try WindowSpec(channels:7, tileHeight:64, tileWidth:64, strideY:48, strideX:48)
        let latentWindow = try WindowSpec(channels:6, tileHeight:64, tileWidth:64, strideY:32, strideX:32)
        let latentCondWindow = try WindowSpec(channels:7, tileHeight:4, tileWidth:4, strideY:1, strideX:1, offsetY:-1, offsetX:-1)
        let decoderWindow = try WindowSpec(channels:2, tileHeight:configuration.decoderTileSize, tileWidth:configuration.decoderTileSize, strideY:configuration.decoderTileStride, strideX:configuration.decoderTileStride)
        let decoderInputWindow = try WindowSpec(channels:6, tileHeight:configuration.decoderTileSize/configuration.latentCompression, tileWidth:configuration.decoderTileSize/configuration.latentCompression, strideY:configuration.decoderTileStride/configuration.latentCompression, strideX:configuration.decoderTileStride/configuration.latentCompression)

        let synthetic = SyntheticConditioningProducer(tileHeight:64, tileWidth:64, config:SyntheticMapConfig(seed: configuration.seed))
        try engine.register(InfiniteTensorStage(tensorID:"conditioning", window:conditioningWindow, valueChannels:5, producer:synthetic))

        let coarseProducer = DenoisingTileProducer(
            denoiser: coarse,
            inputTensorID: nil,
            inputWindow: nil,
            noiseSeed: configuration.seed &+ 1,
            phase: DiffusionPhaseParameters(time: EDMScheduler().trigflowTime(sigma:80), seedOffset:1, name:"coarse"),
            conditioning: [ConditioningDependency(name:"conditioning", tensorID:"conditioning", window:conditioningWindow)],
            legalBatchSizes:[1]
        )
        try engine.register(InfiniteTensorStage(tensorID:"coarse", window:coarseWindow, valueChannels:6, producer:coarseProducer))

        let initLatent = ConsistencyUpdateProducer(
            denoiser: base,
            inputTensorID: nil,
            inputWindow: nil,
            noiseSeed: configuration.seed &+ 5819,
            phase: DiffusionPhaseParameters(time: EDMScheduler().trigflowTime(sigma:80), seedOffset:5819, name:"latent.init"),
            conditioning: [ConditioningDependency(name:"coarse", tensorID:"coarse", window:latentCondWindow)],
            updateTime: EDMScheduler().trigflowTime(sigma:80),
            legalBatchSizes: [1]
        )
        try engine.register(InfiniteTensorStage(tensorID:"latent_init", window:latentWindow, valueChannels:5, producer:initLatent))

        let finalLatentID: String
        if configuration.T == 1 {
            finalLatentID = "latent_init"
        } else {
            let stepLatent = ConsistencyUpdateProducer(
                denoiser: base,
                inputTensorID:"latent_init",
                inputWindow: latentWindow,
                noiseSeed: configuration.seed &+ 5820,
                phase: DiffusionPhaseParameters(time: EDMScheduler.paperLatentIntermediate(), seedOffset:5820, name:"latent.step0"),
                conditioning: [ConditioningDependency(name:"coarse", tensorID:"coarse", window:latentCondWindow)],
                updateTime: EDMScheduler.paperLatentIntermediate(),
                legalBatchSizes: [1]
            )
            try engine.register(InfiniteTensorStage(tensorID:"latents", window:latentWindow, valueChannels:5, producer:stepLatent))
            finalLatentID = "latents"
        }

        let decoderProducer = DecoderTileProducer(denoiser: decoder, latentTensorID: finalLatentID, latentWindow: decoderInputWindow, tileSize: configuration.decoderTileSize, seed: configuration.seed &+ 5819, latentCompression: configuration.latentCompression, phase: DiffusionPhaseParameters(time: EDMScheduler().trigflowTime(sigma:80), seedOffset:5819, name:"decoder"))
        try engine.register(InfiniteTensorStage(tensorID:"residual", window:decoderWindow, valueChannels:1, producer:decoderProducer))
    }

    public func residual(region: Region2D) throws -> MetalTensor {
        try engine.queryNormalized(tensorID:"residual", region:region)
    }

    public func elevation(region: Region2D) throws -> MetalTensor {
        let cb = try context.makeCommandBuffer(label:"terrain.elevation")
        let res = try engine.queryNormalized(tensorID:"residual", region:region, commandBuffer:cb)
        let scaled = try context.allocate(shape:res.shape, scalarType:.float32, storageMode:.storageModePrivate, label:"residual.denorm")
        try engine.kernels.unaryScale(commandBuffer:cb, source:res, destination:scaled, scale:configuration.residualStd, bias:configuration.residualMean)
        let elev = try transforms.inverseSignedSqrt(scaled, commandBuffer:cb)
        try context.runAndWait(cb)
        return elev
    }
}
