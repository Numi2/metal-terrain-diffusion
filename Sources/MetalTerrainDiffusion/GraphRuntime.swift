import Foundation
import Metal

public struct GraphArchive: Codable {
    public struct TensorInfo: Codable { public var name:String; public var shape:[Int]; public var file:String }
    public struct Op: Codable { public var kind:String; public var name:String; public var inputs:[String]; public var outputs:[String]; public var attrs:[String:String]? }
    public var name:String
    public var inputChannels:Int
    public var outputChannels:Int
    public var tileHeight:Int
    public var tileWidth:Int
    public var legalBatchSizes:[Int]
    public var weights:[TensorInfo]
    public var ops:[Op]
}

public final class MetalGraphDenoiser: MetalDenoiser {
    public let name:String; public let inputChannels:Int; public let outputChannels:Int; public let tileHeight:Int; public let tileWidth:Int; public let legalBatchSizes:[Int]
    private let archive: GraphArchive
    private var weights:[String:MetalTensor] = [:]
    private let kernels: KernelEncoder

    public init(context: MetalContext, archiveURL: URL) throws {
        let manifestURL = archiveURL.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let archive = try JSONDecoder().decode(GraphArchive.self, from: manifestData)
        self.archive = archive
        self.name = archive.name
        self.inputChannels = archive.inputChannels
        self.outputChannels = archive.outputChannels
        self.tileHeight = archive.tileHeight
        self.tileWidth = archive.tileWidth
        self.legalBatchSizes = archive.legalBatchSizes
        self.kernels = KernelEncoder(context: context)
        for w in archive.weights {
            guard w.shape.count == 4 || w.shape.count == 2 else { throw TerrainDiffusionError.invalidModelArchive("weight \(w.name) must be 2D or 4D") }
            let url = archiveURL.appendingPathComponent(w.file)
            let data = try Data(contentsOf: url)
            let count = data.count / MemoryLayout<Float>.stride
            let values = data.withUnsafeBytes { raw -> [Float] in
                Array(raw.bindMemory(to: Float.self))
            }
            guard values.count == count else { throw TerrainDiffusionError.invalidModelArchive("bad weight data \(w.name)") }
            let shape: TensorShape
            if w.shape.count == 4 { shape = try TensorShape(n:w.shape[0], c:w.shape[1], h:w.shape[2], w:w.shape[3]) }
            else { shape = try TensorShape(n:1, c:1, h:w.shape[0], w:w.shape[1]) }
            weights[w.name] = try context.upload(values, shape: shape, label: "w.\(w.name)")
        }
    }

    public func encode(_ batch: DenoiseBatch, context: MetalContext) throws -> MetalTensor {
        var table: [String:MetalTensor] = ["input": batch.input]
        for (k,v) in batch.conditioning { table["cond.\(k)"] = v }
        for op in archive.ops {
            switch op.kind {
            case "conv2d":
                let x = try tensor(op.inputs[0], table); let w = try weight(op.inputs[1])
                let cout = intAttr(op,"cout",w.shape.n), kh = intAttr(op,"kh",w.shape.h), kw = intAttr(op,"kw",w.shape.w)
                let sy = intAttr(op,"stride_y",1), sx = intAttr(op,"stride_x",1)
                let oh = intAttr(op,"out_h", x.shape.h / sy), ow = intAttr(op,"out_w", x.shape.w / sx)
                let y = try context.allocate(shape: TensorShape(n:x.shape.n,c:cout,h:oh,w:ow), scalarType:.float32, storageMode:.storageModePrivate, label:op.name)
                try kernels.conv2D(commandBuffer: batch.commandBuffer, source:x, weight:w, destination:y, kernelH:kh, kernelW:kw, groups:intAttr(op,"groups",1), strideY:sy, strideX:sx, padY:intAttr(op,"pad_y",kh/2), padX:intAttr(op,"pad_x",kw/2))
                table[op.outputs[0]] = y
            case "silu", "mp_silu":
                let x = try tensor(op.inputs[0], table); let y=try context.allocate(shape:x.shape, scalarType:.float32, storageMode:.storageModePrivate, label:op.name)
                try kernels.mpSilu(commandBuffer:batch.commandBuffer, source:x, destination:y); table[op.outputs[0]]=y
            case "scale":
                let x=try tensor(op.inputs[0],table); let y=try context.allocate(shape:x.shape, scalarType:.float32, storageMode:.storageModePrivate, label:op.name)
                try kernels.unaryScale(commandBuffer:batch.commandBuffer, source:x, destination:y, scale:floatAttr(op,"scale",1), bias:floatAttr(op,"bias",0)); table[op.outputs[0]]=y
            case "add", "linear_mix":
                let a=try tensor(op.inputs[0],table), b=try tensor(op.inputs[1],table); let y=try context.allocate(shape:a.shape, scalarType:.float32, storageMode:.storageModePrivate, label:op.name)
                try kernels.linearMix(commandBuffer:batch.commandBuffer, a:a, b:b, destination:y, weightA:floatAttr(op,"wa",1), weightB:floatAttr(op,"wb",1)); table[op.outputs[0]]=y
            case "concat_channels":
                let a=try tensor(op.inputs[0],table), b=try tensor(op.inputs[1],table); let y=try context.allocate(shape:TensorShape(n:a.shape.n,c:a.shape.c+b.shape.c,h:a.shape.h,w:a.shape.w), scalarType:.float32, storageMode:.storageModePrivate, label:op.name)
                try kernels.catChannels(commandBuffer:batch.commandBuffer, a:a, b:b, destination:y); table[op.outputs[0]]=y
            case "identity":
                table[op.outputs[0]] = try tensor(op.inputs[0], table)
            default:
                throw TerrainDiffusionError.invalidGraph("Unsupported graph op \(op.kind). Exporter should lower this op before runtime.")
            }
        }
        guard let out = table["output"] ?? table[archive.ops.last?.outputs.first ?? ""] else { throw TerrainDiffusionError.invalidGraph("graph has no output") }
        return out
    }

    private func tensor(_ name:String,_ table:[String:MetalTensor]) throws -> MetalTensor { if let t=table[name]{return t}; throw TerrainDiffusionError.missingTensor(name) }
    private func weight(_ name:String) throws -> MetalTensor { if let t=weights[name]{return t}; throw TerrainDiffusionError.missingTensor("weight \(name)") }
    private func intAttr(_ op:GraphArchive.Op,_ k:String,_ d:Int)->Int{ Int(op.attrs?[k] ?? "") ?? d }
    private func floatAttr(_ op:GraphArchive.Op,_ k:String,_ d:Float)->Float{ Float(op.attrs?[k] ?? "") ?? d }
}
