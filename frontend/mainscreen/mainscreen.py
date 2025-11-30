from kivy.uix.screenmanager import Screen

class HomeScreen(Screen):
    def set_welcome(self, username):
        self.ids.welcome_label.text = f"Chào mừng {username}!"

    def logout(self):
        """Xử lý khi người dùng nhấn nút đăng xuất"""
        app = self.manager.app  # Lấy thể hiện của App
        self.manager.current = "login"  # Quay về màn hình login
        login_screen = self.manager.get_screen("login")
        login_screen.ids.username.text = ""
        login_screen.ids.password.text = ""
        login_screen.ids.message.text = ""

    def infor_user(self):
        user_screen = self.manager.get_screen("user")
        user_screen.show_user_info(
            self.user_info.get("username", ""),
            self.user_info.get("password", ""),
            self.user_info.get("email", "")
        )
        self.manager.current = "user"

    def genAI(self):
        genAi_screen = self.manager.get_screen("genAIscreen")
        self.manager.current = "genAIscreen"

if __name__ == "__main__":
    HomeScreen().run()
