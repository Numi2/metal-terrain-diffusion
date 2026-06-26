import Foundation
import simd

public extension AnunakiTerrainTile {
    func artifactSites(count: Int) -> [SIMD3<Float>] {
        guard resolution > 12 else { return [] }

        let candidates = artifactCandidates()
        var sites: [SIMD3<Float>] = []
        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            guard sites.allSatisfy({ simd_distance($0, candidate.point) > 125 }) else { continue }
            sites.append(candidate.point)
            if sites.count == count { break }
        }
        return sites
    }

    private func artifactCandidates() -> [(score: Float, point: SIMD3<Float>)] {
        var candidates: [(score: Float, point: SIMD3<Float>)] = []

        for z in stride(from: 4, to: resolution - 4, by: 4) {
            for x in stride(from: 4, to: resolution - 4, by: 4) {
                let height = heightAtGrid(x: x, z: z)
                let normal = normalAtGrid(x: x, z: z)
                let score = artifactScore(height: height, slope: 1 - normal.y, curvature: curvatureAtGrid(x: x, z: z))
                let worldX = Float(originX) + Float(x) * spacing
                let worldZ = Float(originZ) + Float(z) * spacing
                candidates.append((score, SIMD3<Float>(worldX, height + 76, worldZ)))
            }
        }

        return candidates
    }

    private func artifactScore(height: Float, slope: Float, curvature: Float) -> Float {
        let basinPenalty: Float = height < 40 ? 0.5 : 1
        return (curvature * 0.12 + slope * 22 + height * 0.025) * basinPenalty
    }

    private func curvatureAtGrid(x: Int, z: Int) -> Float {
        let height = heightAtGrid(x: x, z: z)
        let left = heightAtGrid(x: x - 2, z: z)
        let right = heightAtGrid(x: x + 2, z: z)
        let down = heightAtGrid(x: x, z: z - 2)
        let up = heightAtGrid(x: x, z: z + 2)
        return abs((height * 4) - left - right - down - up)
    }
}
