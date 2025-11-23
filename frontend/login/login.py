from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
import json

# Dữ liệu user (có thể lưu trong users.json)
users = {
    "admin": "123456",
    "bich": "abcd1234"
}
"""with open("users.json") as f:
    users = json.load(f)"""

class LoginScreen(Screen):
    def login(self):
        username = self.ids.username.text
        password = self.ids.password.text

        if username in users and users[username] == password:
            # Lấy instance HomeScreen
            home_screen = self.manager.get_screen("home")
            home_screen.set_welcome(username)
            #self.ids.message.color = (0, 1, 0, 1)  # xanh khi login thành công
            #self.ids.message.text = f"Đăng nhập thành công!"
            self.manager.current = "home"

            home_screen.user_info = {
                "username": username,
                "password": password,
                "email": password
            }

        else:
            self.ids.message.color = (1,0,0,1)
            self.ids.message.text = "Tên đăng nhập hoặc mật khẩu sai!"


class ImageButton(ButtonBehavior, Image):
    pass

class Register(Screen):
    def register(self):
        self.ids.label
        # pass
"""
class LoginApp(App):
    def build(self):
        from kivy.lang import Builder
        Builder.load_file("login.kv")
        return LoginScreen()

if __name__ == "__main__":
    LoginApp().run()"""
