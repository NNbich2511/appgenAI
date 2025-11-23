from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from predict import predict_image

class ImageApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.image = KivyImage()
        self.label = Label(text="Chưa có ảnh nào được chọn.", size_hint=(1, 0.2))

        btn_select = Button(text="Chọn ảnh để nhận dạng", size_hint=(1, 0.2))
        btn_select.bind(on_press=self.choose_image)

        self.layout.add_widget(self.image)
        self.layout.add_widget(self.label)
        self.layout.add_widget(btn_select)
        return self.layout

    def choose_image(self, instance):
        chooser = FileChooserIconView()
        chooser.bind(on_submit=self.load_image)
        self.layout.clear_widgets()
        self.layout.add_widget(chooser)

    def load_image(self, chooser, selection, touch):
        if selection:
            path = selection[0]
            self.image.source = path
            result = predict_image(path)
            self.label.text = f"Kết quả: {result}"
            self.layout.clear_widgets()
            self.layout.add_widget(self.image)
            self.layout.add_widget(self.label)
            btn_back = Button(text="Chọn ảnh khác", size_hint=(1, 0.2))
            btn_back.bind(on_press=self.choose_image)
            self.layout.add_widget(btn_back)

if __name__ == '__main__':
    ImageApp().run()
