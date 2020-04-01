reset
echo "Run this file inside workspace!!"
echo "Compiling..."
echo "Main..."
g++ src/main.cpp -c `pkg-config --cflags --libs opencv4`
echo "Module1..."
g++ src/Module1.cpp -c `pkg-config --cflags --libs opencv4`
echo "Executable"
g++ main.o Module1.o `pkg-config --cflags --libs opencv4`
echo "Done"