//
//  PrePostProcessor.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 6/7/21.
//

// PrePostProcessor is only used for displaying image purposes. The actual algorithm only needs the 800x800
// input image from resizing
import Foundation
import UIKit

struct BoundingBox {
    let x1: Int
    let y1: Int
    let x2: Int
    let y2: Int
}

struct Prediction {
    let classIndex : Int
    let score: Float
    let bbox: BoundingBox
}

extension Prediction {
    static var predictionPreview = [Prediction(classIndex: 13, score: 0.9, bbox: BoundingBox(x1: 25, y1: 25, x2: 100, y2: 100))]
}

class PrePostProcessor : NSObject {
    static let inputWidth = 320
    static let inputHeight = 320

    static let outputColumn = 6 // left, top, right, bot, label, score
    static let threshold : Float = 0.6
    
    static func combineImage(image1: UIImage, background: UIImage) -> UIImage {
        let newSize = CGSize(width: inputWidth, height: inputHeight)
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        
        background.draw(in : CGRect(origin: CGPoint.zero, size: newSize))
        image1.draw(in: CGRect(origin: CGPoint.zero, size: newSize))

        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage!
    }
    
    static func outputsToPredictions(outputs: [NSNumber], imgScaleX: Double, imgScaleY : Double) -> [Prediction] {
        var predictions = [Prediction]()
        for i in 0..<outputs.count / 6 {
            if Float(truncating: outputs[i * outputColumn+4]) > threshold {
                let x1 = Int(imgScaleX * Double(truncating: outputs[i * outputColumn]))
                let y1 = Int(imgScaleY * Double(truncating: outputs[i * outputColumn + 1]))
                let x2 = Int(imgScaleX * Double(truncating: outputs[i * outputColumn + 2]))
                let y2 = Int(imgScaleY * Double(truncating: outputs[i * outputColumn + 3]))
                
                let bbox = BoundingBox(x1: x1, y1: y1, x2: x2, y2: y2)
                //let rect = CGRect(x: x1, y: y1, width: (x2 - x1), height: (y2 - y1))
                let prediction = Prediction(classIndex: Int(truncating: outputs[i*outputColumn+5]), score: Float(truncating: outputs[i*outputColumn+4]), bbox: bbox)
                predictions.append(prediction)
            }
        }
        return predictions
    }
}
