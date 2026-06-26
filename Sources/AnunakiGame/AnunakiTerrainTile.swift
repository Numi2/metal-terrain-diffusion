import Foundation
import simd

public struct AnunakiTerrainTile: Sendable, Hashable {
    public var originX: Int
    public var originZ: Int
    public var resolution: Int
    public var spacing: Float
    public var heights: [Float]
    public var source: AnunakiTerrainSource

    public init(originX: Int, originZ: Int, resolution: Int, spacing: Float, heights: [Float], source: AnunakiTerrainSource) {
        self.originX = originX
        self.originZ = originZ
        self.resolution = max(2, resolution)
        self.spacing = max(1, spacing)
        self.source = source

        let requiredHeightCount = self.resolution * self.resolution
        self.heights = heights.count == requiredHeightCount
            ? heights
            : Array(repeating: 0, count: requiredHeightCount)
    }

    public var worldSize: Float {
        Float(resolution - 1) * spacing
    }

    public func contains(x: Float, z: Float) -> Bool {
        x >= Float(originX)
            && z >= Float(originZ)
            && x <= Float(originX) + worldSize
            && z <= Float(originZ) + worldSize
    }

    public func heightAt(x: Float, z: Float) -> Float {
        let localX = ((x - Float(originX)) / spacing).clamped(to: 0...Float(resolution - 1))
        let localZ = ((z - Float(originZ)) / spacing).clamped(to: 0...Float(resolution - 1))
        let x0 = Int(floor(localX))
        let z0 = Int(floor(localZ))
        let x1 = min(x0 + 1, resolution - 1)
        let z1 = min(z0 + 1, resolution - 1)
        let tx = localX - Float(x0)
        let tz = localZ - Float(z0)

        let h00 = heightAtGrid(x: x0, z: z0)
        let h10 = heightAtGrid(x: x1, z: z0)
        let h01 = heightAtGrid(x: x0, z: z1)
        let h11 = heightAtGrid(x: x1, z: z1)
        return lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), tz)
    }

    public func heightAtGrid(x: Int, z: Int) -> Float {
        heights[z * resolution + x]
    }

    public func normalAtGrid(x: Int, z: Int) -> SIMD3<Float> {
        let x0 = max(0, x - 1)
        let x1 = min(resolution - 1, x + 1)
        let z0 = max(0, z - 1)
        let z1 = min(resolution - 1, z + 1)

        let left = heightAtGrid(x: x0, z: z)
        let right = heightAtGrid(x: x1, z: z)
        let down = heightAtGrid(x: x, z: z0)
        let up = heightAtGrid(x: x, z: z1)
        let normal = SIMD3<Float>(left - right, spacing * 2, down - up)
        let length = simd_length(normal)
        return length > 0 ? normal / length : SIMD3<Float>(0, 1, 0)
    }

    public static func procedural(originX: Int, originZ: Int, resolution: Int, spacing: Float, seed: UInt64) -> AnunakiTerrainTile {
        var heights: [Float] = []
        heights.reserveCapacity(resolution * resolution)

        for zIndex in 0..<resolution {
            for xIndex in 0..<resolution {
                let x = Float(originX) + Float(xIndex) * spacing
                let z = Float(originZ) + Float(zIndex) * spacing
                heights.append(proceduralHeight(x: x, z: z, seed: seed))
            }
        }

        return AnunakiTerrainTile(
            originX: originX,
            originZ: originZ,
            resolution: resolution,
            spacing: spacing,
            heights: heights,
            source: .proceduralFallback
        )
    }

    public static func proceduralHeight(x: Float, z: Float, seed: UInt64) -> Float {
        let seedPhase = Float(seed % 10_000) * 0.001
        let ridge = sin((x + seedPhase * 37) * 0.008) * cos((z - seedPhase * 19) * 0.006)
        let islands = sin((x + z) * 0.0025 + seedPhase) + cos((x - z) * 0.0032 - seedPhase * 0.7)
        let detail = sin(x * 0.031 + seedPhase * 2.1) * sin(z * 0.027 - seedPhase)
        let mask = max(0, islands * 0.5 + 0.48)
        return 18 + mask * mask * 112 + ridge * 34 + detail * 8
    }
}

private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
    a + (b - a) * t
}
