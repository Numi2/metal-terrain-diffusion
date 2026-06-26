import XCTest
import simd
@testable import AnunakiGame

final class AnunakiFlightModelTests: XCTestCase {
    func testBoostConsumesEnergyAndIncreasesSpeed() {
        var normal = AnunakiFlightModel()
        var boosted = AnunakiFlightModel()

        normal.step(input: AnunakiFlightInput(throttle: 1, boost: false), deltaTime: 1 / 30, terrainHeight: 0)
        boosted.step(input: AnunakiFlightInput(throttle: 1, boost: true), deltaTime: 1 / 30, terrainHeight: 0)

        XCTAssertGreaterThan(boosted.state.speed, normal.state.speed)
        XCTAssertLessThan(boosted.state.energy, normal.state.energy)
    }

    func testTerrainCollisionMaintainsMinimumClearance() {
        var model = AnunakiFlightModel(
            state: AnunakiShipState(position: SIMD3<Float>(0, 3, 0), velocity: SIMD3<Float>(8, -90, 0)),
            minimumClearance: 20
        )

        model.step(input: AnunakiFlightInput(throttle: 0), deltaTime: 1 / 30, terrainHeight: 30)

        XCTAssertGreaterThanOrEqual(model.state.position.y, 50)
        XCTAssertGreaterThan(model.state.velocity.y, 0)
        XCTAssertLessThan(model.state.integrity, 1)
    }

    func testTerrainTileInterpolatesHeight() {
        let tile = AnunakiTerrainTile(originX: 0, originZ: 0, resolution: 2, spacing: 10, heights: [0, 10, 20, 30], source: .trainedTerrainDiffuser)

        XCTAssertEqual(tile.heightAt(x: 5, z: 5), 15, accuracy: 0.001)
        XCTAssertEqual(tile.source, .trainedTerrainDiffuser)
    }

    func testArchiveValidationDetectsMissingArchives() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let missing = AnunakiTerrainProvider.missingArchiveNames(at: root)

        XCTAssertEqual(Set(missing), Set(AnunakiTerrainProvider.requiredArchiveNames))
    }

    func testTerrainSourcesHaveDistinctLabels() {
        XCTAssertNotEqual(AnunakiTerrainSource.trainedTerrainDiffuser.rawValue, AnunakiTerrainSource.debugDiffuser.rawValue)
        XCTAssertNotEqual(AnunakiTerrainSource.modelArchivesMissing.rawValue, AnunakiTerrainSource.proceduralFallback.rawValue)
    }

    func testProceduralAdjacentTilesShareBoundaryHeights() {
        let left = AnunakiTerrainTile.procedural(originX: 0, originZ: 0, resolution: 9, spacing: 8, seed: 123)
        let right = AnunakiTerrainTile.procedural(originX: 64, originZ: 0, resolution: 9, spacing: 8, seed: 123)

        for z in 0..<left.resolution {
            let worldZ = Float(z) * left.spacing
            XCTAssertEqual(left.heightAt(x: 64, z: worldZ), right.heightAt(x: 64, z: worldZ), accuracy: 0.001)
        }
    }

    func testArtifactSitesStayAboveTerrain() {
        let tile = AnunakiTerrainTile.procedural(originX: 0, originZ: 0, resolution: 33, spacing: 8, seed: 5819)

        let sites = tile.artifactSites(count: 3)

        XCTAssertFalse(sites.isEmpty)
        for site in sites {
            XCTAssertGreaterThan(site.y, tile.heightAt(x: site.x, z: site.z) + 50)
        }
    }
}
