import pygame
import time
import random

pygame.init()

# Set up the screen
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

# Snake attributes
snake_block = 10
snake_speed = 15

# Snake colors
snake_color = (0, 255, 0)
food_color = (255, 0, 0)
score_color = (0, 0, 0)


# Snake function
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, snake_color, [x[0], x[1], snake_block, snake_block])


# Main game loop
def game_loop():
    game_over = False
    game_close = False

    # Initial snake position
    snake_x = screen_width / 2
    snake_y = screen_height / 2

    # Initial snake movement
    change_x = 0
    change_y = 0

    # Snake body (a list of blocks)
    snake_list = []
    length_of_snake = 1

    # Food position
    food_x = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
    food_y = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0

    # Main game loop
    while not game_over:
        while game_close:
            screen.fill((255, 255, 255))
            font_style = pygame.font.SysFont(None, 50)
            message = font_style.render(
                "You Lost! Press Q-Quit or C-Play Again", True, score_color
            )
            screen.blit(message, [screen_width / 6, screen_height / 3])

            # Display final score
            score = length_of_snake - 1
            score_font = pygame.font.SysFont(None, 35)
            score_message = score_font.render(
                "Your Score: " + str(score), True, score_color
            )
            screen.blit(score_message, [screen_width / 3, screen_height / 2])

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    change_x = -snake_block
                    change_y = 0
                elif event.key == pygame.K_RIGHT:
                    change_x = snake_block
                    change_y = 0
                elif event.key == pygame.K_UP:
                    change_y = -snake_block
                    change_x = 0
                elif event.key == pygame.K_DOWN:
                    change_y = snake_block
                    change_x = 0

        # Boundary conditions (game over when snake hits the screen edge)
        if (
            snake_x >= screen_width
            or snake_x < 0
            or snake_y >= screen_height
            or snake_y < 0
        ):
            game_close = True

        snake_x += change_x
        snake_y += change_y
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, food_color, [food_x, food_y, snake_block, snake_block])
        snake_head = []
        snake_head.append(snake_x)
        snake_head.append(snake_y)
        snake_list.append(snake_head)

        # Adjust snake length when it eats food
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        # Check if snake collides with itself (game over)
        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        # Draw the snake on the screen
        our_snake(snake_block, snake_list)

        # Display the score
        score = length_of_snake - 1
        score_font = pygame.font.SysFont(None, 35)
        score_message = score_font.render(
            "Your Score: " + str(score), True, score_color
        )
        screen.blit(score_message, [0, 0])

        pygame.display.update()

        # If snake eats the food, generate new food position and increase length
        if snake_x == food_x and snake_y == food_y:
            food_x = (
                round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
            )
            food_y = (
                round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0
            )
            length_of_snake += 1

        # Set the speed of the game
        pygame.time.delay(snake_speed)

    pygame.quit()
    quit()


game_loop()
