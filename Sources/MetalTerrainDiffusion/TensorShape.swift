import Foundation
import Metal

public enum ScalarType: String, Codable, Sendable {
    case float32
    case float16
    public var byteCount: Int { self == .float32 ? 4 : 2 }
}

public struct TensorShape: Hashable, Codable, Sendable, CustomStringConvertible {
    public var n: Int
    public var c: Int
    public var h: Int
    public var w: Int
    public init(n: Int = 1, c: Int, h: Int, w: Int) throws {
        guard n > 0, c > 0, h > 0, w > 0 else { throw TerrainDiffusionError.invalidShape("N,C,H,W must be positive; got \(n),\(c),\(h),\(w)") }
        self.n = n; self.c = c; self.h = h; self.w = w
    }
    public var elementCount: Int { n * c * h * w }
    public func byteCount(_ scalarType: ScalarType) -> Int { elementCount * scalarType.byteCount }
    public var description: String { "NCHW(\(n),\(c),\(h),\(w))" }
}

public struct MetalTensor: CustomStringConvertible {
    public let buffer: MTLBuffer
    public let shape: TensorShape
    public let scalarType: ScalarType
    public let byteOffset: Int
    public init(buffer: MTLBuffer, shape: TensorShape, scalarType: ScalarType, byteOffset: Int = 0) {
        self.buffer = buffer; self.shape = shape; self.scalarType = scalarType; self.byteOffset = byteOffset
    }
    public var byteCount: Int { shape.byteCount(scalarType) }
    public var elementCount: Int { shape.elementCount }
    public var description: String { "MetalTensor(\(shape), \(scalarType.rawValue), byteOffset=\(byteOffset))" }
    public func assertSameShape(as other: MetalTensor) throws {
        guard shape == other.shape, scalarType == other.scalarType else { throw TerrainDiffusionError.invalidShape("Expected \(shape)/\(scalarType), got \(other.shape)/\(other.scalarType)") }
    }
}
