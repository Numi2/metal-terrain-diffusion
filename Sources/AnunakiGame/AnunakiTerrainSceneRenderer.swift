#if os(iOS)
import Foundation
import SceneKit
import simd
import UIKit

@MainActor
final class AnunakiTerrainSceneRenderer {
    struct GameplayNodes {
        let artifactRings: [SCNNode]
        let leylineNodes: [SCNNode]

        var allNodes: [SCNNode] {
            artifactRings + leylineNodes
        }
    }

    func makeTerrainNode(_ tile: AnunakiTerrainTile) -> SCNNode {
        let geometry = AnunakiTerrainGeometryFactory.geometry(for: tile)
        return SCNNode(geometry: geometry)
    }

    func makeGameplayNodes(for tile: AnunakiTerrainTile) -> GameplayNodes {
        let sites = tile.artifactSites(count: 4)
        let rings = sites.map(makeArtifactRing(at:))
        let leylines = makeLeylines(connecting: sites)
        return GameplayNodes(artifactRings: rings, leylineNodes: leylines)
    }
}

private extension AnunakiTerrainSceneRenderer {
    func makeArtifactRing(at site: SIMD3<Float>) -> SCNNode {
        let geometry = SCNTorus(ringRadius: 12, pipeRadius: 1.8)
        geometry.firstMaterial?.diffuse.contents = UIColor.systemYellow
        geometry.firstMaterial?.emission.contents = UIColor.systemYellow.withAlphaComponent(0.6)

        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(site.x, site.y, site.z)
        node.name = "artifact"
        return node
    }

    func makeLeylines(connecting sites: [SIMD3<Float>]) -> [SCNNode] {
        guard sites.count > 1 else { return [] }
        return zip(sites.dropLast(), sites.dropFirst()).map(makeLeyline)
    }

    func makeLeyline(from start: SIMD3<Float>, to end: SIMD3<Float>) -> SCNNode {
        let mid = (start + end) * 0.5
        let length = simd_distance(start, end)
        let geometry = SCNCylinder(radius: 1.2, height: CGFloat(length))
        geometry.firstMaterial?.diffuse.contents = UIColor.cyan.withAlphaComponent(0.32)
        geometry.firstMaterial?.emission.contents = UIColor.cyan.withAlphaComponent(0.45)
        geometry.firstMaterial?.isDoubleSided = true

        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(mid.x, mid.y - 35, mid.z)
        node.look(
            at: SCNVector3(end.x, end.y - 35, end.z),
            up: SCNVector3(0, 1, 0),
            localFront: SCNVector3(0, 1, 0)
        )
        return node
    }
}

private enum AnunakiTerrainGeometryFactory {
    static func geometry(for tile: AnunakiTerrainTile) -> SCNGeometry {
        let vertices = makeVertices(for: tile)
        let normals = makeNormals(for: tile)
        let colors = makeColors(for: tile)
        let indices = makeIndices(resolution: tile.resolution)

        let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
        let colorSource = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: colors.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.stride,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride
        )

        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<UInt32>.stride)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )
        let geometry = SCNGeometry(
            sources: [
                SCNGeometrySource(vertices: vertices),
                SCNGeometrySource(normals: normals),
                colorSource,
            ],
            elements: [element]
        )
        configureMaterial(geometry.firstMaterial)
        return geometry
    }

    private static func makeVertices(for tile: AnunakiTerrainTile) -> [SCNVector3] {
        var vertices: [SCNVector3] = []
        vertices.reserveCapacity(tile.resolution * tile.resolution)

        for z in 0..<tile.resolution {
            for x in 0..<tile.resolution {
                let worldX = Float(tile.originX) + Float(x) * tile.spacing
                let worldZ = Float(tile.originZ) + Float(z) * tile.spacing
                vertices.append(SCNVector3(worldX, tile.heights[z * tile.resolution + x], worldZ))
            }
        }
        return vertices
    }

    private static func makeNormals(for tile: AnunakiTerrainTile) -> [SCNVector3] {
        var normals: [SCNVector3] = []
        normals.reserveCapacity(tile.resolution * tile.resolution)

        for z in 0..<tile.resolution {
            for x in 0..<tile.resolution {
                let normal = tile.normalAtGrid(x: x, z: z)
                normals.append(SCNVector3(normal.x, normal.y, normal.z))
            }
        }
        return normals
    }

    private static func makeColors(for tile: AnunakiTerrainTile) -> [SIMD4<Float>] {
        var colors: [SIMD4<Float>] = []
        colors.reserveCapacity(tile.resolution * tile.resolution)

        for z in 0..<tile.resolution {
            for x in 0..<tile.resolution {
                colors.append(tile.colorAtGrid(x: x, z: z))
            }
        }
        return colors
    }

    private static func makeIndices(resolution: Int) -> [UInt32] {
        var indices: [UInt32] = []
        indices.reserveCapacity((resolution - 1) * (resolution - 1) * 6)

        for z in 0..<(resolution - 1) {
            for x in 0..<(resolution - 1) {
                let a = UInt32(z * resolution + x)
                let b = UInt32(z * resolution + x + 1)
                let c = UInt32((z + 1) * resolution + x)
                let d = UInt32((z + 1) * resolution + x + 1)
                indices += [a, c, b, b, c, d]
            }
        }
        return indices
    }

    private static func configureMaterial(_ material: SCNMaterial?) {
        material?.diffuse.contents = UIColor.white
        material?.emission.contents = UIColor(red: 0.00, green: 0.08, blue: 0.07, alpha: 1)
        material?.lightingModel = .physicallyBased
        material?.metalness.contents = 0.04
        material?.roughness.contents = 0.86
    }
}
#endif
