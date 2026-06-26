#if os(iOS)
import SwiftUI

struct AnunakiHUD: View {
    @ObservedObject var controller: AnunakiGameController

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 8) {
                Text(controller.missionText)
                    .font(.system(size: 22, weight: .black, design: .rounded))
                HStack(spacing: 10) {
                    AnunakiMeter(label: "SPD", value: controller.state.speed / 210)
                    AnunakiMeter(label: "ENG", value: controller.state.energy)
                    AnunakiMeter(label: "HUL", value: controller.state.integrity)
                }
                AnunakiTerrainStatus(
                    modelStatus: controller.modelStatus,
                    terrainSource: controller.terrainSource
                )
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(controller.ringsCollected)")
                    .font(.system(size: 30, weight: .black, design: .rounded))
                Text("ARTIFACTS")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
        .foregroundStyle(.white)
        .shadow(color: .black.opacity(0.55), radius: 5, y: 2)
    }
}

private struct AnunakiTerrainStatus: View {
    let modelStatus: AnunakiTerrainSource
    let terrainSource: AnunakiTerrainSource

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(modelStatus.rawValue)
                .font(.system(size: 10, weight: .black, design: .rounded))
                .foregroundStyle(modelStatus == .trainedTerrainDiffuser ? .cyan : .orange)
            if modelStatus == .modelArchivesMissing {
                Text(terrainSource.rawValue)
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundStyle(.cyan.opacity(0.85))
            }
        }
    }
}

private struct AnunakiMeter: View {
    let label: String
    let value: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 9, weight: .bold, design: .rounded))
            GeometryReader { proxy in
                Capsule()
                    .fill(.white.opacity(0.25))
                    .overlay(alignment: .leading) {
                        Capsule()
                            .fill(value > 0.25 ? .cyan : .red)
                            .frame(width: proxy.size.width * CGFloat(value.clamped(to: 0...1)))
                    }
            }
            .frame(width: 62, height: 6)
        }
    }
}
#endif
