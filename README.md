# Сборка проекта

В файле `CMakeLists.txt` укажите пути к Face SDK и OpenCV:
   
Укажите путь к папке Face SDK
   set(BASE_DIR "<путь_к_папке_Face_SDK>")
   
Укажите путь к директории сборки OpenCV
   set(OpenCV_DIR "<путь_к_директории_OpenCV>")
   

Перейдите директорию проекта и выполните команды для создания папки сборки:

mkdir build

cd build

Выполните команду для генерации файлов сборки:

cmake ..

Выполните сборку:

cmake --build .

Скопируйте файл opencv_world4100d.dll из директории:  opencv\build\x64\vc16\bin

в папку сборки: build\Debug


Для запуска выполните например следующую команду из директории build\Debug:

TEST2 --unit_type age --unit_type emotions
