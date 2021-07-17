//
//  DrawingView.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 5/7/21.
//

import Foundation
import SwiftUI
import PencilKit

struct CanvasView : UIViewRepresentable {
    @Binding var isDraw : Bool
    @Binding var canvas : PKCanvasView
    let ink = PKInkingTool(.pen, color: .black)
    let eraser = PKEraserTool(.bitmap)

    
    func makeUIView(context: Context) -> PKCanvasView {
        canvas.drawingPolicy = .anyInput
        canvas.isScrollEnabled = false
        canvas.layer.borderWidth = 10
        canvas.layer.borderColor = UIColor.gray.cgColor
        canvas.tool = isDraw ? ink : eraser
        return canvas
    }

    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {
        uiView.tool = isDraw ? ink : eraser
    }
    
}
