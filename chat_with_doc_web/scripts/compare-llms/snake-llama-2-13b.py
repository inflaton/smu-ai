# Import the necessary libraries
import random
import pygame

# Initialize Pygame
pygame.init()

# Set the screen dimensions
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the title of the window
pygame.display.set_caption("Snake Game")

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Define some constants
BOARD_WIDTH = 20
BOARD_HEIGHT = 20
FOOD_AMOUNT = 5


# Define the snake object
class Snake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.body = [(x, y)]

    def move(self, direction):
        # Move the snake in the specified direction
        if direction == "up":
            self.y -= 1
        elif direction == "down":
            self.y += 1
        elif direction == "left":
            self.x -= 1
        elif direction == "right":
            self.x += 1

        # Check for collisions with the walls or the food
        if self.x < 0 or self.x >= BOARD_WIDTH or self.y < 0 or self.y >= BOARD_HEIGHT:
            print("Game Over!")
            quit()
        elif self.body[-1] != (self.x, self.y):
            print("Collision!")
            quit()


# Define the main game loop
def game_loop():
    # Initialize the snake object
    snake = Snake(BOARD_WIDTH // 2, BOARD_HEIGHT // 2)

    # Initialize the food object
    food = Food(random.randint(0, BOARD_WIDTH), random.randint(0, BOARD_HEIGHT))

    # Game loop
    while True:
        # Clear the screen
        screen.fill(BLACK)

        # Draw the snake
        for segment in snake.body:
            screen.draw.rect(segment, GREEN)

        # Draw the food
        screen.draw.rect(food, RED)

        # Get the user input
        direction = input("Enter a direction (up, down, left, or right): ")

        # Move the snake
        snake.move(direction)

        # Check for collisions
        if snake.body[-1] == food.position:
            print("Yum! You ate the food!")
            food.reset()
            snake.reset()


# Define the food object
class Food:
    def __init__(self, x, y):
        self.position = (x, y)
        self.reset()

    def reset(self):
        # Reset the food position to a random location
        self.position = (
            random.randint(0, BOARD_WIDTH),
            random.randint(0, BOARD_HEIGHT),
        )


# Define the game over function
def game_over():
    # Print the game over message
    print("Game Over!")


# Start the game loop
game_loop()
