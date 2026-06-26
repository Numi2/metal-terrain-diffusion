#if os(iOS)
import QuartzCore
import SceneKit
import simd
import UIKit

@MainActor
final class AnunakiSceneCoordinator: NSObject {
    private let controller: AnunakiGameController
    private let scene = SCNScene()
    private let shipNode = SCNNode()
    private let cameraRig = SCNNode()
    private let cameraNode = SCNNode()
    private let terrainRenderer = AnunakiTerrainSceneRenderer()

    private var displayLink: CADisplayLink?
    private var lastTime: CFTimeInterval = 0
    private var model = AnunakiFlightModel()
    private var provider: AnunakiTerrainProvider?
    private var terrainNodes: [String: SCNNode] = [:]
    private var tiles: [AnunakiTerrainTile] = []
    private var pendingTiles: Set<String> = []
    private var rings: [SCNNode] = []
    private var collected = 0

    init(controller: AnunakiGameController) {
        self.controller = controller
        super.init()
    }

    func makeView() -> SCNView {
        let view = SCNView(frame: .zero)
        view.scene = scene
        view.backgroundColor = AnunakiSceneAssets.backgroundColor
        view.preferredFramesPerSecond = 60
        view.antialiasingMode = .multisampling4X
        view.isPlaying = true

        configureScene()
        view.pointOfView = cameraNode
        startLoop()
        loadTerrainProvider()
        return view
    }

    deinit {
        displayLink?.invalidate()
    }
}

private extension AnunakiSceneCoordinator {
    func configureScene() {
        AnunakiSceneAssets.configureAtmosphere(scene)
        AnunakiSceneAssets.configureCamera(cameraNode, rig: cameraRig, in: scene)
        AnunakiSceneAssets.configureLighting(in: scene)

        shipNode.addChildNode(AnunakiSceneAssets.makeUFO())
        scene.rootNode.addChildNode(shipNode)

        let lookAtShip = SCNLookAtConstraint(target: shipNode)
        lookAtShip.isGimbalLockEnabled = true
        cameraNode.constraints = [lookAtShip]

        AnunakiSceneAssets.addStarfield(to: scene)
        updateScene(deltaTime: 0)
    }

    func startLoop() {
        displayLink?.invalidate()
        let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    func loadTerrainProvider() {
        Task { @MainActor in
            let bootstrap = AnunakiTerrainProvider.archiveFirst(seed: controller.seed)
            provider = bootstrap.provider
            controller.publishModelStatus(bootstrap.status)
            await loadTilesAround(position: model.state.position)
        }
    }

    @objc func tick(_ displayLink: CADisplayLink) {
        if lastTime == 0 {
            lastTime = displayLink.timestamp
            return
        }

        let deltaTime = Float(displayLink.timestamp - lastTime)
        lastTime = displayLink.timestamp
        advanceSimulation(deltaTime: deltaTime)
    }

    func advanceSimulation(deltaTime: Float) {
        let terrainHeight = heightAt(x: model.state.position.x, z: model.state.position.z)
        model.step(input: controller.input, deltaTime: deltaTime, terrainHeight: terrainHeight)
        updateScene(deltaTime: deltaTime)
        collectRings()
        controller.publishFrame(state: model.state, ringsCollected: collected)

        Task { @MainActor in
            await loadTilesAround(position: model.state.position)
        }
    }

    func updateScene(deltaTime: Float) {
        shipNode.position = SCNVector3(model.state.position.x, model.state.position.y, model.state.position.z)
        shipNode.eulerAngles = SCNVector3(model.state.pitch * 0.55, model.state.yaw, -controller.input.yaw * 0.32)
        cameraRig.position = SCNVector3(model.state.position.x, model.state.position.y, model.state.position.z)
        cameraRig.eulerAngles.y = model.state.yaw
        rings.forEach { $0.eulerAngles.y += deltaTime * 1.8 }
    }

    func heightAt(x: Float, z: Float) -> Float {
        if let tile = tiles.first(where: { $0.contains(x: x, z: z) }) {
            return tile.heightAt(x: x, z: z)
        }
        return AnunakiTerrainTile.proceduralHeight(x: x, z: z, seed: controller.seed)
    }
}

private extension AnunakiSceneCoordinator {
    func loadTilesAround(position: SIMD3<Float>) async {
        guard let provider else { return }

        let tileSize = 512
        let centerX = Int(floor(position.x / Float(tileSize))) * tileSize
        let centerZ = Int(floor(position.z / Float(tileSize))) * tileSize

        for dz in -1...1 {
            for dx in -1...1 {
                let originX = centerX + dx * tileSize
                let originZ = centerZ + dz * tileSize
                let key = "\(originX):\(originZ)"
                if terrainNodes[key] != nil || pendingTiles.contains(key) { continue }

                pendingTiles.insert(key)
                let tile = await provider.tile(originX: originX, originZ: originZ)
                addTile(tile, key: key)
                controller.publishTerrainSource(tile.source)
                pendingTiles.remove(key)
            }
        }
    }

    func addTile(_ tile: AnunakiTerrainTile, key: String) {
        let terrainNode = terrainRenderer.makeTerrainNode(tile)
        terrainNodes[key] = terrainNode
        tiles.append(tile)
        scene.rootNode.addChildNode(terrainNode)

        let effects = terrainRenderer.makeGameplayNodes(for: tile)
        rings.append(contentsOf: effects.artifactRings)
        effects.allNodes.forEach(scene.rootNode.addChildNode)
    }

    func collectRings() {
        rings.removeAll { ring in
            let distance = simd_distance(
                SIMD3<Float>(ring.position.x, ring.position.y, ring.position.z),
                model.state.position
            )
            guard distance < 18 else { return false }

            ring.removeFromParentNode()
            collected += 1
            model.state.score += 100
            model.state.energy = min(1, model.state.energy + 0.18)
            return true
        }
    }
}
#endif
