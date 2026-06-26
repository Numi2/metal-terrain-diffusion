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
    print("Generated elevation tensor: \(elev.shape); samples=\(samples)")
} catch {
    fputs("terrain-diffusion-metal failed: \(error)\n", stderr)
    exit(1)
}
