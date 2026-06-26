import Foundation

public struct WindowKey: Hashable, Codable, Sendable, CustomStringConvertible {
    public var tensorID: String
    public var y: Int
    public var x: Int
    public init(tensorID: String, y: Int, x: Int) { self.tensorID = tensorID; self.y = y; self.x = x }
    public var description: String { "\(tensorID)[\(y),\(x)]" }
}

public struct WindowSpec: Hashable, Codable, Sendable, CustomStringConvertible {
    public var channels: Int
    public var tileHeight: Int
    public var tileWidth: Int
    public var strideY: Int
    public var strideX: Int
    public var offsetY: Int
    public var offsetX: Int
    public init(channels: Int, tileHeight: Int, tileWidth: Int, strideY: Int, strideX: Int, offsetY: Int = 0, offsetX: Int = 0) throws {
        guard channels > 0, tileHeight > 0, tileWidth > 0, strideY > 0, strideX > 0 else { throw TerrainDiffusionError.invalidShape("Bad window spec") }
        self.channels = channels; self.tileHeight = tileHeight; self.tileWidth = tileWidth; self.strideY = strideY; self.strideX = strideX; self.offsetY = offsetY; self.offsetX = offsetX
    }
    public var description: String { "WindowSpec(C=\(channels), tile=\(tileHeight)x\(tileWidth), stride=\(strideY)x\(strideX), offset=\(offsetY),\(offsetX))" }
    public func region(for key: WindowKey) -> Region2D { Region2D(y: key.y * strideY + offsetY, x: key.x * strideX + offsetX, height: tileHeight, width: tileWidth) }
    public func localIntersection(windowKey: WindowKey, query: Region2D) -> (windowLocal: Region2D, queryLocal: Region2D)? {
        let wr = region(for: windowKey)
        guard let inter = wr.intersection(query) else { return nil }
        return (Region2D(y: inter.y - wr.y, x: inter.x - wr.x, height: inter.height, width: inter.width), Region2D(y: inter.y - query.y, x: inter.x - query.x, height: inter.height, width: inter.width))
    }
    public func keys(overlapping region: Region2D, tensorID: String) -> [WindowKey] {
        guard !region.isEmpty else { return [] }
        let ky0 = ceilDiv(region.minY - offsetY - tileHeight + 1, strideY)
        let ky1 = floorDiv(region.maxY - 1 - offsetY, strideY)
        let kx0 = ceilDiv(region.minX - offsetX - tileWidth + 1, strideX)
        let kx1 = floorDiv(region.maxX - 1 - offsetX, strideX)
        guard ky1 >= ky0, kx1 >= kx0 else { return [] }
        var out: [WindowKey] = []
        out.reserveCapacity((ky1 - ky0 + 1) * (kx1 - kx0 + 1))
        for yy in ky0...ky1 { for xx in kx0...kx1 { out.append(WindowKey(tensorID: tensorID, y: yy, x: xx)) } }
        return out
    }
}

public struct ParentDependency: Codable, Sendable {
    public var parentTensorID: String
    public var parentWindow: WindowSpec
    public init(parentTensorID: String, parentWindow: WindowSpec) { self.parentTensorID = parentTensorID; self.parentWindow = parentWindow }
}
