//
//  Debugging.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 15/7/21.
//

import Foundation
import SwiftUI
class Debugging {
    
    static func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    
    static func saveVariableString(str: String) {
        let filename = getDocumentsDirectory().appendingPathComponent("output.txt")
        print(filename)
        do {
            try str.write(to: filename, atomically: true, encoding: String.Encoding.utf8)
        } catch {
            // failed to write file – bad permissions, bad filename, missing permissions, or more likely it can't be converted to the encoding
            print("failed")
        }
    }
    
    static func saveImage(image : UIImage) {
        if let data = image.pngData() {
            let filename = getDocumentsDirectory().appendingPathComponent("test.png")
            try? data.write(to: filename)
            print(filename)
        }
    }
    
    static func printTimeElapsedWhenRunningCode(title:String, operation:()->()) {
        let startTime = CFAbsoluteTimeGetCurrent()
        operation()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Time elapsed for \(title): \(timeElapsed) s.")
    }

}
