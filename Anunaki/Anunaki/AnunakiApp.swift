import AnunakiGame
import SwiftUI

@main
struct AnunakiApp: App {
    var body: some Scene {
        WindowGroup {
            AnunakiGameView()
                .preferredColorScheme(.dark)
        }
    }
}
