//
//  ContentView.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 5/7/21.
//

import UIKit

import SwiftUI
import PencilKit

struct ContentView: View {
    @State var isActive : Bool = false
    
    var body: some View {
        NavigationView {
            NavigationLink ("Start Drawing", destination: DrawingView(shouldPopToRootView: self.$isActive), isActive: self.$isActive)
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
}

struct ContentView_Previews: PreviewProvider {
    static var drawing  = PKDrawing()
    static var previews: some View {
        ContentView()
    }
}
