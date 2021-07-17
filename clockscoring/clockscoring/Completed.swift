//
//  Completed.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 5/7/21.
//

import Foundation
import SwiftUI
import PencilKit
struct Completed : View {
    @Binding var shouldPopToRootView : Bool
    @Binding var drawing : UIImage
    @Binding var score : Dictionary<String, Int>
    @Binding var errorMsgs : Dictionary<String, String>
    
    func showErrorMsgs() -> String {
        var errorMsg = ""
        if let contourMsg = self.errorMsgs["contour"] {
            errorMsg += contourMsg
            errorMsg += "\n"
        }
        if let digitMsg = self.errorMsgs["digit"] {
            errorMsg += digitMsg
            errorMsg += "\n"
        }
        if let handMsg = self.errorMsgs["hand"] {
            errorMsg += handMsg
            errorMsg += "\n"
        }
        return errorMsg
    }
    
    var body : some View {
        VStack {
            Spacer()
            Image(uiImage: drawing)
                .resizable()
                .frame(width: 320, height:320)
                .border(/*@START_MENU_TOKEN@*/Color.black/*@END_MENU_TOKEN@*/, width: 2)
            Text("YOUR SCORE").font(.largeTitle)
            Spacer()
            HStack {
                Spacer()
                VStack {
                    Text("Digit").font(.largeTitle)
                    Text(String(score["digit"]!)).font(.largeTitle)
                }
                Spacer()
                VStack {
                    Text("Hand").font(.largeTitle)
                    Text(String(score["hand"]!)).font(.largeTitle)
                }
                Spacer()
                VStack {
                    Text("Contour").font(.largeTitle)
                    Text(String(score["contour"]!)).font(.largeTitle)
                }
                Spacer()
            }
            .padding(.bottom, 300)
            .padding(.leading, 100)
            .padding(.trailing, 100)
            HStack {
                Text(showErrorMsgs()).font(.title)
            }
            .padding(.bottom, 100)
        }
    }
}

struct Completed_Previews: PreviewProvider {
    static var drawing  = UIImage()
    static var shouldPopToRootView = false
    static var score = ["contour" : 0, "digit" : 0, "hand": 0]
    static var errorMsgs = ["contour" : "None", "digit" : "None", "hand": "None"]
    static var previews: some View {
        Completed(shouldPopToRootView : .constant(shouldPopToRootView), drawing: .constant(drawing),
                  score: .constant(score), errorMsgs: .constant(errorMsgs))
    }
}
