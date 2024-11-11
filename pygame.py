import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Window size for the game
WINDOW_SIZE = 800

# Window size for congratulations screen
WIN_WINDOW_SIZE = 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Function to shuffle numbers in the grid
def shuffle_grid(size):
    # Generate a list of numbers from 1 to size^2
    numbers = [i for i in range(1, size * size + 1)]
    # Shuffle the numbers randomly
    random.shuffle(numbers)
    # Create a grid of size*size with the shuffled numbers
    grid = [numbers[i:i+size] for i in range(0, size*size, size)]
    return grid

# Function to draw the grid
def draw_grid(grid, cell_size, font, barriers):
    # Fill the screen with white color
    screen.fill(WHITE)
    # Iterate through each cell in the grid
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            # Draw a rectangle for each cell
            pygame.draw.rect(screen, BLACK, (x * cell_size, y * cell_size, cell_size, cell_size), 2)
            # Render and center the number text for each cell
            number = font.render(str(grid[y][x]), True, BLACK)
            text_rect = number.get_rect(center=(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2))
            screen.blit(number, text_rect)

    # Draw barriers
    for barrier in barriers:
        x1, y1, x2, y2 = barrier
        pygame.draw.line(screen, RED, ((x1 + x2 +1) * cell_size // 2, (y1 + y2) * cell_size // 2),
                         ((x1 + x2 +1) * cell_size // 2, (y1 + y2 ) * cell_size // 2 + cell_size), 5)

# Function to swap two cells in the grid
def swap_cells(grid, y1, x1, y2, x2, barriers):
    # Check if the swap is allowed based on barriers
    if (x1, y1, x2, y2) in barriers or (x2, y2, x1, y1) in barriers:
        return
    # Swap the values of two cells in the grid
    grid[y1][x1], grid[y2][x2] = grid[y2][x2], grid[y1][x1]

# Function to create random barriers
def create_barriers(size):
    barriers = []
    while len(barriers) < size - 1:  # n-1 barriers for an nxn grid
        # Randomly choose coordinates for the barrier
        x1, y1 = random.randint(0, size - 2), random.randint(0, size - 1)
        x2, y2 = x1 + 1, y1
        if (x1, y1, x2, y2) not in barriers and (x2, y2, x1, y1) not in barriers:
            barriers.append((x1, y1, x2, y2))
    return barriers

# Main game function
def main(size, cell_size):
    # Initialize font for displaying numbers
    font = pygame.font.Font(None, 50)
    # Shuffle the grid
    grid = shuffle_grid(size)
    # Create random barriers
    barriers = create_barriers(size)
    # Initialize game variables
    running = True
    selected_cell = None
    moves = 0

    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                clicked_col = x // cell_size
                clicked_row = y // cell_size
                if selected_cell is None:
                    selected_cell = (clicked_row, clicked_col)
                else:
                    target_cell = (clicked_row, clicked_col)
                    if abs(selected_cell[0] - target_cell[0]) + abs(selected_cell[1] - target_cell[1]) == 1:
                        swap_cells(grid, selected_cell[0], selected_cell[1], target_cell[0], target_cell[1], barriers)
                        moves += 1
                        # Check if the grid is ordered
                        if all(grid[i][j] == i * size + j + 1 for i in range(size) for j in range(size)):
                            running = False
                    selected_cell = None

        # Draw the grid, barriers, and update the display
        draw_grid(grid, cell_size, font, barriers)
        pygame.display.flip()

    # Display congratulations screen
    win_screen = pygame.Surface((WIN_WINDOW_SIZE, WIN_WINDOW_SIZE))
    win_screen.fill(WHITE)
    win_font = pygame.font.Font(None, 25)

    win_message = "Congratulations! Number of moves: {}".format(moves)
    win_text = win_font.render(win_message, True, BLACK)
    win_rect = win_text.get_rect(center=(WIN_WINDOW_SIZE // 2, WIN_WINDOW_SIZE // 2))
    win_screen.blit(win_text, win_rect)

    win_window = pygame.display.set_mode((WIN_WINDOW_SIZE, WIN_WINDOW_SIZE))
    pygame.display.set_caption("Congratulations!")
    win_window.blit(win_screen, (0, 0))
    pygame.display.flip()

    # Wait for 5 seconds before quitting
    pygame.time.wait(5000)
    pygame.quit()

# Start the game with the selected difficulty
def start_game(difficulty):
    if difficulty == "Easy":
        main(3, WINDOW_SIZE // 3)
    elif difficulty == "Medium":
        main(4, WINDOW_SIZE // 4)
    elif difficulty == "Hard":
        main(5, WINDOW_SIZE // 5)

# Function to display the difficulty selection screen
def show_difficulty_select():
    # Fill the screen with white color
    screen.fill(WHITE)
    # Initialize font for displaying difficulty options
    font = pygame.font.Font(None, 50)

    # Render and center the "Easy" text
    easy_text = font.render("Easy", True, BLACK)
    easy_rect = easy_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 4))
    screen.blit(easy_text, easy_rect)

    # Render and center the "Medium" text
    medium_text = font.render("Medium", True, BLACK)
    medium_rect = medium_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    screen.blit(medium_text, medium_rect)

    # Render and center the "Hard" text
    hard_text = font.render("Hard", True, BLACK)
    hard_rect = hard_text.get_rect(center=(WINDOW_SIZE // 2, 3 * WINDOW_SIZE // 4))
    screen.blit(hard_text, hard_rect)

    pygame.display.flip()

    # Wait for user input to select the difficulty
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y < WINDOW_SIZE // 3:
                    return "Easy"
                elif y < 2 * WINDOW_SIZE // 3:
                    return "Medium"
                else:
                    return "Hard"

# Start the game
if __name__ == "__main__":
    # Set up the game window and caption
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Puzzle Game")

    # Display the difficulty selection screen
    difficulty = show_difficulty_select()
    # Start the game with the selected difficulty
    start_game(difficulty)
