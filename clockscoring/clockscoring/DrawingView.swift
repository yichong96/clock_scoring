//
//  DrawingView.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 5/7/21.
//

import Foundation
import SwiftUI
import PencilKit
import UIKit

struct DrawingView : View {
    // @State var drawing : PKDrawing = PKDrawing()
    @State var canvas : PKCanvasView = PKCanvasView()
    @State var isDraw : Bool = true
    @State var showLoading = false
    var shouldPopToRootView : Binding<Bool>
    @State var score : Dictionary<String, Int> = ["contour": 0, "digit": 0, "hand": 0]
    @State var completedImage : UIImage = UIImage()
    @State var isDoneComputingScore = false
    @State var errorMsgs : Dictionary<String, String> = [String : String]()
    var inferencer = ObjectDetector()
    @State var pixelBuffer : [Float32] = []
        
    private func runInference() {
        let startTime = CFAbsoluteTimeGetCurrent()
        //perform action
        showLoading = true
        var bounds = self.canvas.drawing.bounds
        bounds = bounds.insetBy(dx: CGFloat(-20), dy: CGFloat(-20))
        let canvasImage = self.canvas.drawing.image(from: bounds, scale : 1.0)  // get canvas image of arbitrary size

        let resizedImage = canvasImage.resized(to: CGSize(width: CGFloat(PrePostProcessor.inputWidth), height: CGFloat(PrePostProcessor.inputHeight)))  // resize the image to specified width and height
        let whiteImage = UIImage(color: .white, size: CGSize(width: PrePostProcessor.inputWidth, height: PrePostProcessor.inputHeight))!    // define white background of same width and height as resizedImage
        completedImage = PrePostProcessor.combineImage(image1: resizedImage, background: whiteImage)    // overlay image onto white background
        
        self.pixelBuffer = completedImage.normalized()!
        
        Debugging.saveImage(image: completedImage)
        DispatchQueue.global().async {
            guard let outputs = self.inferencer.module.detect(image: &self.pixelBuffer) else {
                return
            }
            let predictions = PrePostProcessor.outputsToPredictions(outputs: outputs, imgScaleX: 1.0, imgScaleY: 1.0)
            print(predictions)
            var time = [String : Int]()
            time["minute"] = 10
            time["hour"] = 11
            (score, errorMsgs) = Scoring.score(predictions: predictions, time: time)
            completedImage = completedImage.addBoundingBoxImage(predictions: predictions)
            DispatchQueue.main.async {
                self.isDoneComputingScore = true
                self.showLoading = false
                self.canvas.drawing = PKDrawing()   // clear drawing
                self.pixelBuffer = []   // clear pixel buffer
                let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("Time elapsed: \(timeElapsed) s.")
            }
        }
    }
    
    var body: some View {
        ZStack {
            VStack {
                Button(action: {
                    isDraw.toggle()
                }) {
                    Text("Toggle pencil/eraser").font(.largeTitle)
                }.padding(.trailing, 10)
                CanvasView(isDraw: $isDraw, canvas: $canvas)
                HStack {
                    NavigationLink(destination: Completed(shouldPopToRootView: self.shouldPopToRootView, drawing : self.$completedImage, score: $score, errorMsgs: $errorMsgs),
                        isActive: $isDoneComputingScore)
                        {
                        Button(action: {
                                runInference()
                        }) {
                            Text("Submit Drawing").bold()
                            }
                        }
                        .padding()
                        .background(Color(red: 0, green : 0.6 , blue : 0))
                        .clipShape(Capsule())
                        .foregroundColor(.white)
                }
            }
            if (showLoading) {
                ProgressView("Loading")
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(CGSize(width: 2.0, height: 2.0))
            }
        }
    }
}

struct DrawingView_Previews: PreviewProvider {
    static var drawing = PKDrawing()
    static var canvasImage = UIImage()
    static var shouldPopToRootView = false
    static var previews: some View {
        DrawingView(shouldPopToRootView : .constant(shouldPopToRootView))
    }
}
