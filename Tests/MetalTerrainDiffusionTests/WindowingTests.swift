import XCTest
@testable import MetalTerrainDiffusion

final class WindowingTests: XCTestCase {
    func testOverlappingKeysForNegativeCoordinates() throws {
        let spec = try WindowSpec(channels: 2, tileHeight: 64, tileWidth: 64, strideY: 32, strideX: 32)
        let keys = spec.keys(overlapping: Region2D(y: -5, x: -5, height: 10, width: 10), tensorID: "x")
        XCTAssertFalse(keys.isEmpty)
        XCTAssertTrue(keys.contains(WindowKey(tensorID: "x", y: -1, x: -1)))
    }
}
