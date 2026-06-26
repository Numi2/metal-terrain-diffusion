import Foundation
import simd

public extension AnunakiTerrainTile {
    func colorAtGrid(x: Int, z: Int) -> SIMD4<Float> {
        let normal = normalAtGrid(x: x, z: z)
        let height = heightAtGrid(x: x, z: z)
        let slope = 1 - max(0, min(1, normal.y))
        let high = smoothstep(104, 150, height)
        let low = 1 - smoothstep(42, 82, height)
        let mineral = max(0, sin(Float(x) * 0.61 + Float(z) * 0.37) * 0.5 + 0.5) * slope

        var rgb = lerp(SIMD3<Float>(0.08, 0.23, 0.20), SIMD3<Float>(0.12, 0.12, 0.16), slope)
        rgb = lerp(rgb, SIMD3<Float>(0.58, 0.66, 0.70), high * 0.65)
        rgb = lerp(rgb, SIMD3<Float>(0.06, 0.16, 0.24), low * 0.55)
        rgb = lerp(rgb, SIMD3<Float>(0.02, 0.78, 0.72), mineral * 0.28)
        return SIMD4<Float>(rgb.x, rgb.y, rgb.z, 1)
    }
}

private func lerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

private func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
    let t = ((x - edge0) / (edge1 - edge0)).clamped(to: 0...1)
    return t * t * (3 - 2 * t)
}
