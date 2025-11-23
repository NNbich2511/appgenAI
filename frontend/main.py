from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder
import os

# Import các màn hình
from mainscreen.mainscreen import HomeScreen
from login.login import LoginScreen
from user.user import User

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.app = self  # Cho phép Screen truy cập App thông qua self.manager.app
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(User(name="user"))
        return sm

# Load file KV
kv_path_login = os.path.join(os.path.dirname(__file__), "login", "login.kv")
kv_path_home = os.path.join(os.path.dirname(__file__), "mainscreen", "mainscreen.kv")
kv_path_user = os.path.join(os.path.dirname(__file__), "user", "user.kv")
Builder.load_file(kv_path_login)
Builder.load_file(kv_path_home)
Builder.load_file(kv_path_user)

if __name__ == "__main__":
    MyApp().run()
