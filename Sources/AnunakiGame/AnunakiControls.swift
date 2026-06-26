#if os(iOS)
import SwiftUI

struct AnunakiControls: View {
    @ObservedObject var controller: AnunakiGameController
    @State private var throttle: Float = 0.72

    var body: some View {
        HStack(alignment: .bottom) {
            AnunakiJoystick(onChanged: controller.setStick)
            Spacer()
            VStack(spacing: 14) {
                Button(action: controller.pulseTractorBeam) {
                    Image(systemName: "scope")
                        .font(.system(size: 22, weight: .bold))
                        .frame(width: 56, height: 56)
                }
                .buttonStyle(AnunakiGlassButtonStyle())

                Button(action: {}) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 25, weight: .bold))
                        .frame(width: 68, height: 68)
                }
                .buttonStyle(AnunakiPressButtonStyle(onChanged: controller.setBoost))

                Slider(
                    value: Binding(
                        get: { Double(throttle) },
                        set: updateThrottle
                    ),
                    in: 0...1
                )
                .frame(width: 128)
                .tint(.cyan)
            }
        }
        .foregroundStyle(.white)
    }

    private func updateThrottle(_ value: Double) {
        throttle = Float(value)
        controller.setThrottle(throttle)
    }
}

private struct AnunakiJoystick: View {
    let onChanged: (CGVector) -> Void
    @State private var offset: CGSize = .zero

    var body: some View {
        Circle()
            .fill(.black.opacity(0.25))
            .frame(width: 124, height: 124)
            .overlay {
                Circle()
                    .stroke(.white.opacity(0.28), lineWidth: 2)
                Circle()
                    .fill(.cyan.opacity(0.82))
                    .frame(width: 48, height: 48)
                    .offset(offset)
            }
            .gesture(DragGesture(minimumDistance: 0).onChanged(handleDrag).onEnded(endDrag))
    }

    private func handleDrag(_ value: DragGesture.Value) {
        let radius: CGFloat = 48
        let dx = value.translation.width.clamped(to: -radius...radius)
        let dy = value.translation.height.clamped(to: -radius...radius)
        offset = CGSize(width: dx, height: dy)
        onChanged(CGVector(dx: dx / radius, dy: dy / radius))
    }

    private func endDrag(_ value: DragGesture.Value) {
        offset = .zero
        onChanged(.zero)
    }
}

private struct AnunakiGlassButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(.black.opacity(configuration.isPressed ? 0.44 : 0.24), in: Circle())
            .overlay(Circle().stroke(.white.opacity(0.26), lineWidth: 1))
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
    }
}

private struct AnunakiPressButtonStyle: ButtonStyle {
    let onChanged: (Bool) -> Void

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(configuration.isPressed ? .cyan.opacity(0.78) : .black.opacity(0.28), in: Circle())
            .overlay(Circle().stroke(.white.opacity(0.28), lineWidth: 1))
            .scaleEffect(configuration.isPressed ? 0.92 : 1)
            .onChange(of: configuration.isPressed) { _, isPressed in
                onChanged(isPressed)
            }
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
#endif
