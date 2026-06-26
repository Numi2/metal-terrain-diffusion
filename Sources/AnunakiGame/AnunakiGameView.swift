#if os(iOS)
import SwiftUI

public struct AnunakiGameView: View {
    @StateObject private var controller: AnunakiGameController

    public init(seed: UInt64 = 0xA11A_AA1E_5EED) {
        _controller = StateObject(wrappedValue: AnunakiGameController(seed: seed))
    }

    public var body: some View {
        ZStack {
            AnunakiSceneView(controller: controller)
                .ignoresSafeArea()

            VStack {
                AnunakiHUD(controller: controller)
                Spacer()
                AnunakiControls(controller: controller)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
        }
        .statusBarHidden()
    }
}
#endif
