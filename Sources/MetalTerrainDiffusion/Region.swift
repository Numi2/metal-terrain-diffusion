import Foundation

@inline(__always) public func floorDiv(_ a: Int, _ b: Int) -> Int {
    precondition(b > 0)
    var q = a / b
    let r = a % b
    if r != 0 && ((r > 0) != (b > 0)) { q -= 1 }
    return q
}

@inline(__always) public func ceilDiv(_ a: Int, _ b: Int) -> Int {
    -floorDiv(-a, b)
}

public struct Region2D: Hashable, Codable, Sendable, CustomStringConvertible {
    public var y: Int
    public var x: Int
    public var height: Int
    public var width: Int
    public init(y: Int, x: Int, height: Int, width: Int) {
        self.y = y; self.x = x; self.height = max(height, 0); self.width = max(width, 0)
    }
    public var minY: Int { y }
    public var minX: Int { x }
    public var maxY: Int { y + height }
    public var maxX: Int { x + width }
    public var isEmpty: Bool { height <= 0 || width <= 0 }
    public var description: String { "Region(y:\(y), x:\(x), h:\(height), w:\(width))" }
    public func intersection(_ other: Region2D) -> Region2D? {
        let yy0 = max(minY, other.minY), xx0 = max(minX, other.minX)
        let yy1 = min(maxY, other.maxY), xx1 = min(maxX, other.maxX)
        guard yy1 > yy0, xx1 > xx0 else { return nil }
        return Region2D(y: yy0, x: xx0, height: yy1 - yy0, width: xx1 - xx0)
    }
}
