import numpy as np
import random
import matplotlib.pyplot as plt
import time  # Для замера времени выполнения

from data import Matrix, coordinates


class AntColonyTSP:
    def __init__(self, distance_matrix, coordinates, num_ants, num_iterations, alpha=1.0, beta=2.0,
                 evaporation_rate=0.3):
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.num_cities = len(distance_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Параметр влияния феромона
        self.beta = beta  # Параметр влияния видимости (1 / расстояние)
        self.evaporation_rate = evaporation_rate  # Коэффициент испарения
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))  # Инициализация феромона

    def run(self, show_plot=False, info_interval=10):
        best_path = None
        best_path_length = float('inf')

        start_time = time.time()  # Начало замера времени выполнения

        for iteration in range(self.num_iterations):
            all_paths = self.construct_solutions()
            self.update_pheromones(all_paths)

            # Проверка на лучшее решение в текущей итерации
            for path, path_length in all_paths:
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            # Вывод информации и графика раз в info_interval итераций или на последней итерации
            if (iteration + 1) % info_interval == 0 or iteration == 0 or iteration == (num_iterations - 1):
                print(f"Итерация {iteration + 1}, Лучший путь: {best_path_length}")
                if show_plot:
                    self.plot_solution(best_path, iteration + 1)

        execution_time = time.time() - start_time  # Конец замера времени выполнения

        return best_path, best_path_length, execution_time

    def construct_solutions(self):
        all_paths = []
        for ant in range(self.num_ants):
            path = [random.randint(0, self.num_cities - 1)]
            while len(path) < self.num_cities:
                current_city = path[-1]
                next_city = self.choose_next_city(path, current_city)
                path.append(next_city)
            path_length = self.calculate_path_length(path)
            all_paths.append((path, path_length))
        return all_paths

    def choose_next_city(self, path, current_city):
        probabilities = []
        for city in range(self.num_cities):
            if city not in path:
                pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
                visibility = (1.0 / self.distance_matrix[current_city][city]) ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)

        probabilities_sum = sum(probabilities)
        probabilities = [p / probabilities_sum for p in probabilities]

        return random.choices(range(self.num_cities), probabilities)[0]

    def calculate_path_length(self, path):
        total_distance = 0
        for i in range(len(path)):
            total_distance += self.distance_matrix[path[i - 1]][path[i]]
        return total_distance

    def update_pheromones(self, all_paths):
        # Испарение феромона
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # Добавление феромона
        for path, path_length in all_paths:
            for i in range(len(path)):
                current_city = path[i - 1]
                next_city = path[i]
                self.pheromone_matrix[current_city][next_city] += 1.0 / path_length
                self.pheromone_matrix[next_city][current_city] += 1.0 / path_length

    def plot_solution(self, best_path, iteration):
        """Визуализация текущего лучшего решения."""
        complete_path = best_path + [best_path[0]]
        path_coordinates = [self.coordinates[i] for i in complete_path]

        x, y = zip(*path_coordinates)

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, marker='o', linestyle='-', label=f'Итерация {iteration}')
        plt.title(f'Лучший найденный путь на итерации {iteration}')
        plt.xlabel('Координата X')
        plt.ylabel('Координата Y')
        plt.grid()
        plt.scatter(*zip(*self.coordinates), c='red', s=50)
        plt.legend()
        plt.show()


# Функция для проведения серии экспериментов
def run_experiments(parameters):
    results = []
    for num_ants, num_iterations, alpha, beta, evaporation_rate in parameters:
        print(f"\nЭксперимент: num_ants={num_ants}, num_iterations={num_iterations}, "
              f"alpha={alpha}, beta={beta}, evaporation_rate={evaporation_rate}")

        # Создание объекта класса Matrix
        matrix_data = Matrix()
        distance_matrix = matrix_data.distance_matrix

        # Запуск муравьиного алгоритма
        ant_colony = AntColonyTSP(distance_matrix, coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate)
        best_path, best_path_length, execution_time = ant_colony.run(show_plot=False, info_interval=num_iterations)

        # Сохранение результатов
        results.append({
            'num_ants': num_ants,
            'num_iterations': num_iterations,
            'alpha': alpha,
            'beta': beta,
            'evaporation_rate': evaporation_rate,
            'best_path_length': best_path_length,
            'best_path': best_path,
            'execution_time': execution_time
        })

        # Вывод результатов эксперимента
        print(f"Время выполнения: {execution_time:.4f} секунд")
        print(f"Длина найденного пути: {best_path_length}")
        print(f"Лучший путь: {best_path}")

    return results



# Создание объекта класса Matrix
matrix_data = Matrix()
distance_matrix = matrix_data.distance_matrix

# Параметры муравьиного алгоритма
num_ants = 30
num_iterations = 100

# Запуск муравьиного алгоритма
ant_colony = AntColonyTSP(distance_matrix, coordinates, num_ants, num_iterations, evaporation_rate=0.3)
best_path, best_path_length, execution_time = ant_colony.run(show_plot=True, info_interval=25)

print(f"\nЛучший найденный путь: {best_path}")
print(f"Длина пути: {best_path_length}")
print(f"Время выполнения: {execution_time:.4f} секунд")


# Примеры параметров для экспериментов
experiment_parameters = [
    (30, 100, 1.0, 2.0, 0.3),
    (50, 100, 1.0, 2.0, 0.5),
    (30, 150, 0.5, 2.5, 0.2),
    (40, 120, 1.5, 1.5, 0.4),
    (25, 80, 1.0, 2.0, 0.6),
]

# Запуск серии экспериментов
experiment_results = run_experiments(experiment_parameters)

# Вывод суммарных результатов
for idx, result in enumerate(experiment_results):
    print(f"\nРезультаты эксперимента {idx + 1}:")
    print(f"Параметры: num_ants={result['num_ants']}, num_iterations={result['num_iterations']}, "
          f"alpha={result['alpha']}, beta={result['beta']}, evaporation_rate={result['evaporation_rate']}")
    print(f"Найденный путь: {result['best_path']}")
    print(f"Длина пути: {result['best_path_length']}")
    print(f"Время выполнения: {result['execution_time']:.4f} секунд")
