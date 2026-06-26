import Foundation
import MetalTerrainDiffusion

public actor AnunakiTerrainProvider {
    public static let requiredArchiveNames = [
        "coarse_model.metalgraph",
        "base_model.metalgraph",
        "decoder_model.metalgraph",
    ]

    private let pipeline: TerrainDiffusionPipeline?
    private let seed: UInt64
    private let diffusionSource: AnunakiTerrainSource
    private var cache: [TileKey: AnunakiTerrainTile] = [:]

    public init(
        seed: UInt64 = 0xA11A_AA1E_5EED,
        pipeline: TerrainDiffusionPipeline? = nil,
        diffusionSource: AnunakiTerrainSource = .trainedTerrainDiffuser
    ) {
        self.seed = seed
        self.pipeline = pipeline
        self.diffusionSource = pipeline == nil ? .proceduralFallback : diffusionSource
    }

    public static func missingArchiveNames(at root: URL) -> [String] {
        requiredArchiveNames.filter { archiveName in
            let manifest = root.appendingPathComponent(archiveName).appendingPathComponent("manifest.json")
            return !FileManager.default.fileExists(atPath: manifest.path)
        }
    }

    public static func archiveFirst(seed: UInt64 = 0xA11A_AA1E_5EED, bundle: Bundle = .main) -> AnunakiTerrainBootstrap {
        let candidates = archiveCandidates(bundle: bundle)

        for root in candidates where missingArchiveNames(at: root).isEmpty {
            if let bootstrap = trainedBootstrap(seed: seed, archiveRoot: root) {
                return bootstrap
            }
        }

        if let debugProvider = try? debugDiffusion(seed: seed) {
            return AnunakiTerrainBootstrap(provider: debugProvider, status: .modelArchivesMissing, archiveRoot: nil)
        }
        return AnunakiTerrainBootstrap(provider: AnunakiTerrainProvider(seed: seed), status: .proceduralFallback, archiveRoot: nil)
    }

    public static func debugDiffusion(seed: UInt64 = 0xA11A_AA1E_5EED) throws -> AnunakiTerrainProvider {
        var configuration = TerrainDiffusionConfiguration()
        configuration.seed = seed
        configuration.decoderTileSize = 128
        configuration.decoderTileStride = 96

        let pipeline = TerrainDiffusionPipeline(context: try MetalContext(), configuration: configuration)
        try pipeline.buildDebugIdentityStages()
        return AnunakiTerrainProvider(seed: seed, pipeline: pipeline, diffusionSource: .debugDiffuser)
    }

    public func tile(originX: Int, originZ: Int, resolution: Int = 65, spacing: Float = 8) async -> AnunakiTerrainTile {
        let tileRequest = TileRequest(originX: originX, originZ: originZ, resolution: resolution, spacing: spacing)
        let key = TileKey(tileRequest)
        if let cached = cache[key] { return cached }

        let tile = makeDiffusionTile(tileRequest) ?? AnunakiTerrainTile.procedural(
            originX: tileRequest.alignedOriginX,
            originZ: tileRequest.alignedOriginZ,
            resolution: tileRequest.resolution,
            spacing: tileRequest.spacing,
            seed: seed
        )
        cache[key] = tile
        evictOldestTileIfNeeded()
        return tile
    }
}

private extension AnunakiTerrainProvider {
    static func archiveCandidates(bundle: Bundle) -> [URL] {
        var candidates: [URL] = []
        let fileManager = FileManager.default
        if let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            candidates.append(documents.appendingPathComponent("AnunakiModels"))
        }
        if let bundleRoot = bundle.resourceURL?.appendingPathComponent("AnunakiModels") {
            candidates.append(bundleRoot)
        }
        return candidates
    }

    static func trainedBootstrap(seed: UInt64, archiveRoot: URL) -> AnunakiTerrainBootstrap? {
        do {
            var configuration = TerrainDiffusionConfiguration()
            configuration.seed = seed
            configuration.decoderTileSize = 512
            configuration.decoderTileStride = 384

            let context = try MetalContext()
            let pipeline = try TerrainDiffusionPipeline.fromArchives(context: context, root: archiveRoot, configuration: configuration)
            let provider = AnunakiTerrainProvider(seed: seed, pipeline: pipeline, diffusionSource: .trainedTerrainDiffuser)
            return AnunakiTerrainBootstrap(provider: provider, status: .trainedTerrainDiffuser, archiveRoot: archiveRoot)
        } catch {
            return nil
        }
    }

    func makeDiffusionTile(_ request: TileRequest) -> AnunakiTerrainTile? {
        guard let pipeline else { return nil }

        do {
            let sampleStride = max(1, Int(request.spacing.rounded()))
            let diffusionExtent = (request.resolution - 1) * sampleStride + 1
            let region = Region2D(
                y: request.alignedOriginZ,
                x: request.alignedOriginX,
                height: diffusionExtent,
                width: diffusionExtent
            )
            let tensor = try pipeline.elevation(region: region)
            let raw = try pipeline.context.download(tensor)
            let heights = decimate(raw, resolution: request.resolution, sampleStride: sampleStride, diffusionExtent: diffusionExtent)
            return AnunakiTerrainTile(
                originX: request.alignedOriginX,
                originZ: request.alignedOriginZ,
                resolution: request.resolution,
                spacing: request.spacing,
                heights: heights,
                source: diffusionSource
            )
        } catch {
            return nil
        }
    }

    func decimate(_ values: [Float], resolution: Int, sampleStride: Int, diffusionExtent: Int) -> [Float] {
        var heights: [Float] = []
        heights.reserveCapacity(resolution * resolution)

        for zIndex in 0..<resolution {
            for xIndex in 0..<resolution {
                let rawIndex = zIndex * sampleStride * diffusionExtent + xIndex * sampleStride
                heights.append(shapeDiffusionHeight(values[rawIndex]))
            }
        }
        return heights
    }

    func evictOldestTileIfNeeded() {
        guard cache.count > 48 else { return }
        let oldestKey = cache.keys.sorted { lhs, rhs in
            lhs.x == rhs.x ? lhs.z < rhs.z : lhs.x < rhs.x
        }.first
        if let oldestKey {
            cache.removeValue(forKey: oldestKey)
        }
    }
}

private struct TileRequest {
    let alignedOriginX: Int
    let alignedOriginZ: Int
    let resolution: Int
    let spacing: Float

    init(originX: Int, originZ: Int, resolution: Int, spacing: Float) {
        self.resolution = max(2, resolution)
        self.spacing = max(1, spacing)
        let tileSize = Int(Float(self.resolution - 1) * self.spacing)
        alignedOriginX = Self.align(originX, to: tileSize)
        alignedOriginZ = Self.align(originZ, to: tileSize)
    }

    private static func align(_ value: Int, to size: Int) -> Int {
        guard size > 0 else { return value }
        return floorDiv(value, size) * size
    }
}

private struct TileKey: Hashable {
    let x: Int
    let z: Int
    let resolution: Int
    let spacing: Int

    init(_ request: TileRequest) {
        x = request.alignedOriginX
        z = request.alignedOriginZ
        resolution = request.resolution
        spacing = Int(request.spacing)
    }
}

private func shapeDiffusionHeight(_ value: Float) -> Float {
    guard value.isFinite else { return 0 }
    return 58 + tanh(value * 0.025) * 82
}
