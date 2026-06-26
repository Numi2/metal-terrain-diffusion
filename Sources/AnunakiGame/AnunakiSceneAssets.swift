#if os(iOS)
import SceneKit
import UIKit

enum AnunakiSceneAssets {
    static let backgroundColor = UIColor(red: 0.02, green: 0.05, blue: 0.09, alpha: 1)

    static func configureAtmosphere(_ scene: SCNScene) {
        scene.fogStartDistance = 900
        scene.fogEndDistance = 2_200
        scene.fogColor = backgroundColor
    }

    static func configureCamera(_ camera: SCNNode, rig: SCNNode, in scene: SCNScene) {
        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 66
        camera.camera?.zNear = 1
        camera.camera?.zFar = 3_200
        camera.position = SCNVector3(0, 54, 132)
        rig.addChildNode(camera)
        scene.rootNode.addChildNode(rig)
    }

    static func configureLighting(in scene: SCNScene) {
        let sun = SCNNode()
        sun.light = SCNLight()
        sun.light?.type = .directional
        sun.light?.intensity = 1_100
        sun.eulerAngles = SCNVector3(-0.75, 0.55, 0.15)
        scene.rootNode.addChildNode(sun)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.color = UIColor(red: 0.24, green: 0.34, blue: 0.48, alpha: 1)
        ambient.light?.intensity = 350
        scene.rootNode.addChildNode(ambient)
    }

    static func makeUFO() -> SCNNode {
        let root = SCNNode()
        root.addChildNode(makeSaucer())
        root.addChildNode(makeDome())
        root.addChildNode(makeTractorBeam())
        root.addChildNode(makeEngineLight())
        return root
    }

    static func addStarfield(to scene: SCNScene) {
        for index in 0..<110 {
            scene.rootNode.addChildNode(makeStar(index: index))
        }
    }
}

private extension AnunakiSceneAssets {
    static func makeSaucer() -> SCNNode {
        let geometry = SCNCylinder(radius: 12, height: 3.4)
        geometry.radialSegmentCount = 72
        geometry.firstMaterial?.diffuse.contents = UIColor(white: 0.72, alpha: 1)
        geometry.firstMaterial?.metalness.contents = 0.8
        geometry.firstMaterial?.roughness.contents = 0.28
        return SCNNode(geometry: geometry)
    }

    static func makeDome() -> SCNNode {
        let geometry = SCNSphere(radius: 6.2)
        geometry.segmentCount = 48
        geometry.firstMaterial?.diffuse.contents = UIColor.cyan.withAlphaComponent(0.55)
        geometry.firstMaterial?.emission.contents = UIColor.cyan.withAlphaComponent(0.28)

        let node = SCNNode(geometry: geometry)
        node.scale = SCNVector3(1, 0.46, 1)
        node.position.y = 2.6
        return node
    }

    static func makeTractorBeam() -> SCNNode {
        let geometry = SCNCone(topRadius: 2, bottomRadius: 15, height: 38)
        geometry.firstMaterial?.diffuse.contents = UIColor.cyan.withAlphaComponent(0.18)
        geometry.firstMaterial?.isDoubleSided = true

        let node = SCNNode(geometry: geometry)
        node.position.y = -22
        node.opacity = 0.34
        return node
    }

    static func makeEngineLight() -> SCNNode {
        let node = SCNNode()
        node.light = SCNLight()
        node.light?.type = .omni
        node.light?.color = UIColor.cyan
        node.light?.intensity = 850
        node.light?.attenuationEndDistance = 180
        node.position = SCNVector3(0, -5, 0)
        return node
    }

    static func makeStar(index: Int) -> SCNNode {
        let angle = Float(index) * 2.399963
        let radius = Float(1_300 + (index % 17) * 54)
        let y = Float(220 + (index * 37) % 520)

        let geometry = SCNSphere(radius: CGFloat(0.8 + Float(index % 5) * 0.16))
        geometry.firstMaterial?.emission.contents = UIColor(white: 0.72 + CGFloat(index % 4) * 0.06, alpha: 1)
        geometry.firstMaterial?.diffuse.contents = UIColor.clear

        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(cos(angle) * radius, y, sin(angle) * radius)
        return node
    }
}
#endif
