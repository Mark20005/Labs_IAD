import lab_1
import lab_2
import lab_3
import lab_4


def show_menu():
    print('------Labs------')
    print('1. Lab 1')
    print('2. Lab 2')
    print('3. Lab 3')
    print('4. Lab 4')
    print('0. Exit')


while True:
    show_menu()
    choose = int(input('Choose your lab (0 to exit): '))

    if choose == 0:
        print("Exiting the program.")
        break
    elif choose == 1:
        lab_1.run_lab()  # Викликає функцію run_lab() зі скрипту lab_1.py
    elif choose == 2:
        lab_2.run_lab()  # Викликає функцію run_lab() зі скрипту lab_2.py
    elif choose == 3:
        lab_3.run_lab()  # Викликає функцію run_lab() зі скрипту lab_3.py
    elif choose == 4:
        lab_4.run_lab()  # Викликає функцію run_lab() зі скрипту lab_4.py
    else:
        print("Invalid choice, please choose a number between 0 and 4.")

