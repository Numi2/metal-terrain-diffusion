#if os(iOS)
import SceneKit
import SwiftUI

struct AnunakiSceneView: UIViewRepresentable {
    @ObservedObject var controller: AnunakiGameController

    func makeCoordinator() -> AnunakiSceneCoordinator {
        AnunakiSceneCoordinator(controller: controller)
    }

    func makeUIView(context: Context) -> SCNView {
        context.coordinator.makeView()
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
#endif
