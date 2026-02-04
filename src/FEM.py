import numpy as np
from numba import njit
from config import *
import matplotlib.pyplot as plt
import pygame 
import Newmark

def FEM_setup(WIDTH, HEIGHT):
    gamma = np.linspace(-1,1,100000)
    dgamma = gamma[1]-gamma[0]
    number_of_elements = 80
    number_of_nodes = number_of_elements*3-(number_of_elements-1)
    epsilon = 1
    mu = 1
    L = 10
    h = L/number_of_elements
    c = 1/np.sqrt(epsilon*mu)


    N = 1/2 * np.array([gamma**2 - gamma, 2*(1-gamma**2), gamma**2 + gamma])
    B = 1/2 * 2/h * np.array([2*gamma-1,-4*gamma, 2*gamma + 1])

    K_dummy = np.zeros((3,3,np.size(gamma)))

    for i in range(np.size(gamma)):
        K_dummy[:,:,i] =c**2 * np.outer(B[:,i],B[:,i])

    K_element = h/2 * np.trapezoid(K_dummy,axis=2,dx=dgamma)
    M_dummy = np.zeros((3,3,np.size(gamma)))

    for i in range(np.size(gamma)):
        M_dummy[:,:,i] = np.outer(N[:,i],N[:,i])

    M_element = h/2 * mu*epsilon*np.trapezoid(M_dummy,axis=2,dx=dgamma)
    K_global = np.zeros((number_of_nodes,number_of_nodes))
    M_global = np.zeros_like(K_global)
    for i in range(number_of_elements):
        K_global[2*i:2*i+3,i*2:i*2+3] += K_element
        M_global[2*i:2*i+3,i*2:i*2+3] += M_element

    C = 0.0001*K_global+0.0001*M_global

    t = np.linspace(0,100,10000)
    dt = t[1]-t[0]
    f = np.zeros((number_of_nodes,np.size(t)))

    f[number_of_nodes//2, :] = 500 * np.exp(-((t-5)/0.5)**2)

    u0 = v0 = np.zeros(number_of_nodes)

    M_global_cal = M_global[1:-1,1:-1]
    K_global_cal = K_global[1:-1,1:-1]
    f_cal = f[1:-1,:]
    C_global_cal = C[1:-1,1:-1]
    u0_cal = u0[1:-1]
    v0_cal = v0[1:-1]

    u,v,a = Newmark.linear_newmark_krenk(M_global_cal,C_global_cal,K_global_cal,f_cal,u0_cal,v0_cal,dt)

    return u, number_of_nodes

def FEM_draw(screen, frame, u, number_of_nodes, x_pixels, SCREEN_WIDTH, SCREEN_HEIGHT, y_scale, nt):
    y_center = (SCREEN_HEIGHT - MENU_HIGHT_MULTI*SCREEN_HEIGHT) // 2
    # undeformed axis (x-axis)
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[0], y_center),
        (x_pixels[-1], y_center),
        1
    )
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[-1], y_center),
        (x_pixels[-1]-10, y_center-10),
        1
    )
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[-1], y_center),
        (x_pixels[-1]-10, y_center+10),
        1
    )
    font = pygame.font.SysFont(None, 24)
    text_surf_x = font.render('x', True, BLACK)
    text_rect_x = text_surf_x.get_rect(center=(x_pixels[-1]+15, y_center-15))

    screen.blit(text_surf_x, text_rect_x)

    
    # undeformed axis (y-axis)
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[0], 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT),
        (x_pixels[0], SCREEN_HEIGHT - 1.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT),
        1
    )
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[0], 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT),
        (x_pixels[0]+10, 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT + 10),
        1
    )
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[0], 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT),
        (x_pixels[0]-10, 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT + 10),
        1
    )
    text_surf_y = font.render('y', True, BLACK)
    text_rect_y = text_surf_y.get_rect(center=(x_pixels[0]-20, 0.5*MENU_HIGHT_MULTI*SCREEN_HEIGHT))

    screen.blit(text_surf_y, text_rect_y)

    # deformed shape
    y_disp = y_center - y_scale * u[:, frame]

    # draw elements
    for i in range(number_of_nodes - 3):
        pygame.draw.line(
            screen,
            (0, 200, 255),
            (x_pixels[i], y_disp[i]),
            (x_pixels[i + 1], y_disp[i + 1]),
            2
        )

    # draw nodes
    for i in range(number_of_nodes-2):
        pygame.draw.circle(
            screen,
            (255, 100, 100),
            (int(x_pixels[i]), int(y_disp[i])),
            3
        )
    # Amplitude marker
    pygame.draw.line(
        screen,
        (80, 80, 80),
        (x_pixels[0]-10, y_disp[(number_of_nodes-2)//2]),
        (x_pixels[0]+10, y_disp[(number_of_nodes-2)//2]),
        1
    )
    frame = (frame + 1) % nt

