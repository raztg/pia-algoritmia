#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct vector2 {
    double x;
    double y;
};

double distance(struct vector2 a, struct vector2 b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

double tour_length(struct vector2 coordinates[], int coordinates_length) {
    double result = 0;
    for (int i = 0; i < coordinates_length - 1; ++i) {
        result = result + distance(coordinates[i], coordinates[i + 1]);
    }
    result = result + distance(coordinates[coordinates_length - 1], coordinates[0]);
    return result;
}

int main(void) {
    srand(time(NULL));
    FILE *train;
    train = fopen("hh_tsp_dataset/data/train.csv", "r");
    if (train == NULL) {
        printf("Can't open file\n");
        return 1;
    }
    else {
        printf("instance_id,n,distribution,mean_pairwise_dist,std_coords,mean_centroid_dist,mean_nn_dist,cost_hill_climbing_2opt,cost_simulated_annealing_2opt,cost_mst_2opt,best_algorithm\n");
        char train_line_buffer[256];
        while (fgets(train_line_buffer, 256, train)) {
            char inst_buffer[10];
            for (int i = 0; train_line_buffer[i] != ','; ++i) {
                inst_buffer[i] = train_line_buffer[i];
            }
            inst_buffer[9] = '\0';
            if (inst_buffer[8] == '_') {
                continue;
            }
            char inst_path[256];
            snprintf(inst_path, 256, "hh_tsp_dataset/data/instances/%s.csv", inst_buffer);
            FILE *inst;
            inst = fopen(inst_path, "r");
            if (inst == NULL) {
                printf("Can't open file %s\n", inst_path);
                return 1;
            }
            else {
                struct vector2 coordinates[100];
                int coordinates_length = 0;
                /* BEGIN INPUT */
                {
                    char inst_line_buffer[256];
                    while (fgets(inst_line_buffer, 256, inst)) {
                        char x_buffer[30];
                        int x_buffer_length = 0;
                        char y_buffer[30];
                        int y_buffer_length = 0;
                        int y_switch = 0;
                        for (int i = 0; inst_line_buffer[i] != '\n' && inst_line_buffer[i] != '\r' && inst_line_buffer[i] != '\0'; ++i) {
                            if (!y_switch) {
                                if (inst_line_buffer[i] != ',') {
                                    x_buffer[x_buffer_length] = inst_line_buffer[i];
                                    x_buffer_length = x_buffer_length + 1;
                                }
                                else {
                                    y_switch = 1;
                                }
                            }
                            else {
                                y_buffer[y_buffer_length] = inst_line_buffer[i];
                                y_buffer_length = y_buffer_length + 1;
                            }
                        }
                        x_buffer[x_buffer_length] = '\0';
                        y_buffer[y_buffer_length] = '\0';
                        if (x_buffer[0] == 'x') {
                            continue;
                        }
                        coordinates[coordinates_length].x = atof(x_buffer);
                        coordinates[coordinates_length].y = atof(y_buffer);
                        coordinates_length = coordinates_length + 1;
                    }
                    fclose(inst);
                }
                struct vector2 nearest_neighbor_tour[100];
                int nearest_neighbor_tour_length = 0;
                /* BEGIN NEAREST_NEIGHBOR */
                {
                    struct vector2 unvisited[100];
                    int unvisited_length = 0;
                    for (int i = 0; i < coordinates_length; ++i) {
                        unvisited[i] = coordinates[i];
                        unvisited_length = unvisited_length + 1;
                    }
                    struct vector2 current = unvisited[0];
                    for (int i = 0; i < unvisited_length - 1; ++i) {
                        unvisited[i] = unvisited[i + 1];
                    }
                    unvisited_length = unvisited_length - 1;
                    nearest_neighbor_tour[nearest_neighbor_tour_length] = current;
                    nearest_neighbor_tour_length = nearest_neighbor_tour_length + 1;
                    while (unvisited_length > 0) {
                        int next = 0;
                        double minimum = distance(current, unvisited[next]);
                        for (int i = 1; i < unvisited_length; ++i) {
                            if (distance(current, unvisited[i]) < minimum) {
                                next = i;
                                minimum = distance(current, unvisited[next]);
                            }
                        }
                        current = unvisited[next];
                        for (int i = next; i < unvisited_length - 1; ++i) {
                            unvisited[i] = unvisited[i + 1];
                        }
                        unvisited_length = unvisited_length - 1;
                        nearest_neighbor_tour[nearest_neighbor_tour_length] = current;
                        nearest_neighbor_tour_length = nearest_neighbor_tour_length + 1;
                    }
                }
                struct vector2 tour1[100];
                int tour1_length = 0;
                /* BEGIN HILL_CLIMBING */
                {
                    for (int i = 0; i < nearest_neighbor_tour_length; ++i) {
                        tour1[i] = nearest_neighbor_tour[i];
                        tour1_length = tour1_length + 1;
                    }
                    int improved = 1;
                    while (improved) {
                        improved = 0;
                        for (int i = 0; i < tour1_length - 1; ++i) {
                            for (int j = i + 2; j < tour1_length; ++j) {
                                if (i == 0 && j == tour1_length - 1) {
                                    continue;
                                }
                                double delta = distance(tour1[i], tour1[j]) + distance(tour1[i + 1], tour1[(j + 1) % tour1_length]) - distance(tour1[i], tour1[i + 1]) - distance(tour1[j], tour1[(j + 1) % tour1_length]);
                                if (delta < -0.000000000001) {
                                    int left = i + 1;
                                    int right = j;
                                    while (left < right) {
                                        struct vector2 swap = tour1[left];
                                        tour1[left] = tour1[right];
                                        tour1[right] = swap;
                                        left = left + 1;
                                        right = right - 1;
                                    }
                                    improved = 1;
                                }
                            }
                        }
                    }
                }
                struct vector2 tour2[100];
                int tour2_length = 0;
                /* BEGIN SIMULATED_ANNEALING */
                {
                    for (int i = 0; i < nearest_neighbor_tour_length; ++i) {
                        tour2[i] = nearest_neighbor_tour[i];
                        tour2_length = tour2_length + 1;
                    }
                    struct vector2 best_tour[100];
                    int best_tour_length = 0;
                    for (int i = 0; i < tour2_length; ++i) {
                        best_tour[i] = tour2[i];
                        best_tour_length = best_tour_length + 1;
                    }
                    double current_cost = tour_length(tour2, tour2_length);
                    double best_cost = current_cost;
                    int time = 0;
                    double temperature = 1000.0;
                    while (1) {
                        time = time + 1;
                        temperature = temperature * 0.9995;
                        if (time > 10000) {
                            break;
                        }
                        int i = rand() % (tour2_length - 1);
                        int j = rand() % (tour2_length - 1 - i) + i + 1;
                        if (i == 0 && j == tour2_length - 1) {
                            continue;
                        }
                        double delta = distance(tour2[i], tour2[j]) + distance(tour2[i + 1], tour2[(j + 1) % tour2_length]) - distance(tour2[i], tour2[i + 1]) - distance(tour2[j], tour2[(j + 1) % tour2_length]);
                        if (delta < -0.000000000001 || (double) rand() / RAND_MAX < exp(-1 * delta / temperature)) {
                            int left = i + 1;
                            int right = j;
                            while (left < right) {
                                struct vector2 swap = tour2[left];
                                tour2[left] = tour2[right];
                                tour2[right] = swap;
                                left = left + 1;
                                right = right - 1;
                            }
                            current_cost = current_cost + delta;
                            if (current_cost < best_cost) {
                                best_cost = current_cost;
                                for (int k = 0; k < tour2_length; ++k) {
                                    best_tour[k] = tour2[k];
                                }
                            }
                        }
                    }
                    for (int i = 0; i < best_tour_length; ++i) {
                        tour2[i] = best_tour[i];
                    }
                }
                struct vector2 tour3[100];
                int tour3_length = 0;
                /* BEGIN MST */
                {
                    int in_tree[100];
                    int in_tree_length = 0;
                    double min_edge[100];
                    int min_edge_length = 0;
                    int parent[100];
                    int parent_length = 0;
                    for (int i = 0; i < coordinates_length; ++i) {
                        in_tree[i] = 0;
                        in_tree_length = in_tree_length + 1;
                        min_edge[i] = HUGE_VAL;
                        min_edge_length = min_edge_length + 1;
                        parent[i] = -1;
                        parent_length = parent_length + 1;
                    }
                    min_edge[0] = 0.0;
                    int adj[100][100];
                    int adj_lengths[100] = {0};
                    for (int i = 0; i < coordinates_length; ++i) {
                        int u = -1;
                        double minimum = HUGE_VAL;
                        for (int v = 0; v < coordinates_length; ++v) {
                            if (!in_tree[v]) {
                                if (min_edge[v] < minimum) {
                                    u = v;
                                    minimum = min_edge[v];
                                }
                            }
                        }
                        if (u == -1) {
                            break;
                        }
                        in_tree[u] = 1;
                        if (parent[u] != -1) {
                            adj[u][adj_lengths[u]] = parent[u];
                            adj_lengths[u] = adj_lengths[u] + 1;
                            adj[parent[u]][adj_lengths[parent[u]]] = u;
                            adj_lengths[parent[u]] = adj_lengths[parent[u]] + 1;
                        }
                        for (int v = 0; v < coordinates_length; ++v) {
                            if (!in_tree[v]) {
                                double d = distance(coordinates[u], coordinates[v]);
                                if (d < min_edge[v]) {
                                    min_edge[v] = d;
                                    parent[v] = u;
                                }
                            }
                        }
                    }
                    int visited[100];
                    int visited_length = 0;
                    for (int i = 0; i < coordinates_length; ++i) {
                        visited[i] = 0;
                        visited_length = visited_length + 1;
                    }
                    int stack[100];
                    int stack_top = -1;
                    stack[++stack_top] = 0;
                    while (stack_top >= 0) {
                        int current = stack[stack_top--];
                        if (!visited[current]) {
                            visited[current] = 1;
                            tour3[tour3_length] = coordinates[current];
                            tour3_length = tour3_length + 1;
                            for (int i = adj_lengths[current] - 1; i >= 0; --i) {
                                int neighbor = adj[current][i];
                                if (!visited[neighbor]) {
                                    stack[++stack_top] = neighbor;
                                }
                            }
                        }
                    }
                }
                /* BEGIN OUTPUT */
                {
                    double tour1_final = tour_length(tour1, tour1_length);
                    double tour2_final = tour_length(tour2, tour2_length);
                    double tour3_final = tour_length(tour3, tour3_length);
                    const char *best_algorithm = "hill_climbing_2opt";
                    double best_tour_final = tour1_final;
                    if (tour2_final < best_tour_final) {
                        best_tour_final = tour2_final;
                        best_algorithm = "simulated_annealing_2opt";
                    }
                    if (tour3_final < best_tour_final) {
                        best_tour_final = tour3_final;
                        best_algorithm = "mst_2opt";
                    }
                    char result_buffer[256];
                    snprintf(result_buffer, 256, ",%f,%f,%f,%s\n", tour1_final, tour2_final, tour3_final, best_algorithm);
                    char final_buffer[512];
                    int final_buffer_length = 0;
                    for (int i = 0; i < 256; ++i) {
                        if (train_line_buffer[i] != '\n' && train_line_buffer[i] != '\r' && train_line_buffer[i] != '\0') {
                            final_buffer[final_buffer_length] = train_line_buffer[i];
                            final_buffer_length = final_buffer_length + 1;
                        }
                        else {
                            break;
                        }
                    }
                    for (int i = 0; i < 256; ++i) {
                        if (result_buffer[i] != '\n' && result_buffer[i] != '\r' && result_buffer[i] != '\0') {
                            final_buffer[final_buffer_length] = result_buffer[i];
                            final_buffer_length = final_buffer_length + 1;
                        }
                        else {
                            break;
                        }
                    }
                    final_buffer[final_buffer_length] = '\0';
                    final_buffer_length = final_buffer_length + 1;
                    printf("%s\n", final_buffer);
                }
            }
        }
        fclose(train);
    }
    return 0;
}
