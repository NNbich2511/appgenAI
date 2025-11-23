from kivy.uix.screenmanager import Screen

class User(Screen):
    def show_user_info(self, username, password, email):
        self.ids.name_username.text = f"ID: {username}"
        self.ids.name_password.text = f"Pass: {password}"
        self.ids.email.text = f"Email: {email}"



"""if __name__ == "__main__":
    User().run()"""
"""
import webbrowser
webbrowser.open("https://www.google.com")"""