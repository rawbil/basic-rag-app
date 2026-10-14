#/bin/sh
sudo apt install cowsay -y
cowsay -f dragon "Rawr! Run, I am a dragon!" >> dragon.txt
grep -i "dragon" dragon.txt
cat dragon.txt
ls -ltra