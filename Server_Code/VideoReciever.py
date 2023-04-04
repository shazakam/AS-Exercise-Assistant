import pyrebase


class VideoMessenger:
    def __init__(self):
        self.firebaseConfig = {
        "apiKey": "AIzaSyBy4flGunTpgc_G6yDQofsfxrFDYcg9U0A",
        "authDomain": "dissdatabase-2ee66.firebaseapp.com",
        "databaseURL": "https://dissdatabase-2ee66-default-rtdb.europe-west1.firebasedatabase.app",
        "projectId": "dissdatabase-2ee66",
        "storageBucket": "dissdatabase-2ee66.appspot.com",
        "messagingSenderId": "1029847685739",
        "appId": "1:1029847685739:web:979041a293ecd4cc5f6f2e",
        "measurementId": "G-4VJCBVRVYJ"
        }
        self.firebase = pyrebase.initialize_app(self.firebaseConfig)
        self.storage = self.firebase.storage()
        
   
    def recieveVideo(self):
        
        self.storage.child("Videos/myVideo.mp4").download("myVideo.mp4")
        
    def sendVideo(self):
        self.storage.child("Videos/myVideo.mp4").put("myVideo1.mp4")