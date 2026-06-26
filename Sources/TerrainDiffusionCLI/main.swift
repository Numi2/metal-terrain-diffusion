import Foundation
import MetalTerrainDiffusion

struct Arguments {
    var modelRoot: URL?
    var cacheRoot: URL?
    var x = 0
    var y = 0
    var width = 512
    var height = 512
    var seed: UInt64 = 0x1234_5678_9abc_def0
    var debugIdentity = false
    var validateArchives = false
    var minVariance: Float = 0
    var requireFinite = false
}

func parse() -> Arguments {
    var a = Arguments()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        switch k {
        case "--models": if let v = it.next() { a.modelRoot = URL(fileURLWithPath: v) }
        case "--cache": if let v = it.next() { a.cacheRoot = URL(fileURLWithPath: v) }
        case "--x": if let v = it.next(), let n = Int(v) { a.x = n }
        case "--y": if let v = it.next(), let n = Int(v) { a.y = n }
        case "--width": if let v = it.next(), let n = Int(v) { a.width = n }
        case "--height": if let v = it.next(), let n = Int(v) { a.height = n }
        case "--seed": if let v = it.next(), let n = UInt64(v) { a.seed = n }
        case "--debug-identity": a.debugIdentity = true
        case "--validate-archives": a.validateArchives = true
        case "--min-variance": if let v = it.next(), let n = Float(v) { a.minVariance = n }
        case "--require-finite": a.requireFinite = true
        default: break
        }
    }
    return a
}

let args = parse()
do {
    let context = try MetalContext()
    var cfg = TerrainDiffusionConfiguration()
    cfg.seed = args.seed
    let pipeline: TerrainDiffusionPipeline
    if args.debugIdentity || args.modelRoot == nil {
        pipeline = TerrainDiffusionPipeline(context: context, configuration: cfg, tileStore: args.cacheRoot.flatMap { try? PersistentTileStore(context: context, root: $0) })
        try pipeline.buildDebugIdentityStages()
    } else {
        pipeline = try TerrainDiffusionPipeline.fromArchives(context: context, root: args.modelRoot!, configuration: cfg, persistentCache: args.cacheRoot)
    }
    let region = Region2D(y: args.y, x: args.x, height: args.height, width: args.width)
    let elev = try pipeline.elevation(region: region)
    let values = try context.download(elev)
    let samples = values.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ",")
    if args.validateArchives {
        let finite = values.filter(\.isFinite)
        let minValue = finite.min() ?? .nan
        let maxValue = finite.max() ?? .nan
        let mean = finite.reduce(Float(0), +) / Float(max(finite.count, 1))
        let variance = finite.reduce(Float(0)) { $0 + pow($1 - mean, 2) } / Float(max(finite.count, 1))
        let source = args.debugIdentity || args.modelRoot == nil ? "debug-identity" : "metalgraph-archives"
        print(String(format: "Validation source=%@ shape=%@ min=%.4f max=%.4f mean=%.4f variance=%.4f finite=%d/%d samples=%@", source, String(describing: elev.shape), minValue, maxValue, mean, variance, finite.count, values.count, samples))
        if args.requireFinite && finite.count != values.count {
            fputs("terrain-diffusion-metal validation failed: non-finite values present\n", stderr)
            exit(2)
        }
        if variance < args.minVariance {
            fputs(String(format: "terrain-diffusion-metal validation failed: variance %.6f < required %.6f\n", variance, args.minVariance), stderr)
            exit(2)
        }
    } else {
        print("Generated elevation tensor: \(elev.shape); samples=\(samples)")
    }
} catch {
    fputs("terrain-diffusion-metal failed: \(error)\n", stderr)
    exit(1)
}
