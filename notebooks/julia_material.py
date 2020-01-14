import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import bqplot

def representation_complexe():
    real = widgets.BoundedFloatText(
        value=1,
        min=-2.0,
        max=2.0,
        step=0.1,
        disabled=True
    )

    imag = widgets.BoundedFloatText(
        value=1,
        min=-2.0,
        max=2.0,
        step=0.1,
        disabled=True
    )
    
    sc_x = bqplot.LinearScale(min=-2, max=2)
    sc_y = bqplot.LinearScale(min=-2, max=2)
    ax_x = bqplot.Axis(scale=sc_x, offset=dict(value=0.5), grid_lines='none')
    ax_y = bqplot.Axis(scale=sc_y, orientation='vertical', offset=dict(value=0.5), grid_lines='none')

    z_point = bqplot.Scatter(x=[real.value], y=[imag.value], scales={'x': sc_x, 'y': sc_y}, colors=['green'],
                   enable_move=True)
    z_point.update_on_move = True

    fig = bqplot.Figure(marks=[z_point], axes=[ax_x, ax_y],
             min_aspect_ratio=1, max_aspect_ratio=1)

    complex_z = widgets.HBox([widgets.Label('$z = $'), real, widgets.Label('$ + $'), imag, widgets.Label('$i$')])

    def update_z(change=None):
        real.value = z_point.x[0]
        imag.value = z_point.y[0]
    
    z_point.observe(update_z, names=['x', 'y'])
    
    #def update_point(change=None):
    #    z_point.x = [real.value]
    #    z_point.y = [imag.value]

    #real.observe(update_point, names='value')
    #imag.observe(update_point, names='value')
    
    return widgets.VBox([fig, complex_z], layout=widgets.Layout(align_items="center"))

def orbit(z, c, eps=1e-6, lim=1e5):
    out = [z]
    ite = 0
    while eps < np.abs(out[-1]) < lim and ite < 30:
        out.append(out[-1]**2 + c)
        ite += 1
    return np.asarray(out)

def plot_orbit(z0=0.5+0.5*1j, c=0+0*1j):
    sc_x = bqplot.LinearScale(min=-1.2, max=1.2)
    sc_y = bqplot.LinearScale(min=-1.2, max=1.2)

    c_point = bqplot.Scatter(x=[c.real], y=[c.imag], scales={'x': sc_x, 'y': sc_y}, colors=['red'],
                   enable_move=True, default_size=200)
    c_point.update_on_move = True

    z_point = bqplot.Scatter(x=[z0.real], y=[z0.imag], scales={'x': sc_x, 'y': sc_y}, colors=['green'],
                   enable_move=True, default_size=200)
    z_point.update_on_move = True

    scatt = bqplot.Scatter(x=[], y=[], scales={'x': sc_x, 'y': sc_y}, colors=['black'], default_size=20)

    theta = np.linspace(0, 2.*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)
    circle = bqplot.Lines(x=x, y=y, scales={'x': sc_x, 'y': sc_y}, colors=['black'])
    lin = bqplot.Lines(x=[], y=[], scales={'x': sc_x, 'y': sc_y}, colors=['black'], stroke_width=1)

    def update_line(change=None):
        out = orbit(z_point.x + 1j*z_point.y, c_point.x + 1j*c_point.y)
        lin.x = out.real
        lin.y = out.imag
        scatt.x = out.real.flatten()
        scatt.y = out.imag.flatten()

    update_line()
    # update line on change of x or y of scatter
    c_point.observe(update_line, names=['x'])
    c_point.observe(update_line, names=['y'])
    z_point.observe(update_line, names=['x'])
    z_point.observe(update_line, names=['y'])
    ax_x = bqplot.Axis(scale=sc_x, offset=dict(value=0.5), grid_lines='none')
    ax_y = bqplot.Axis(scale=sc_y, orientation='vertical', offset=dict(value=0.5), grid_lines='none')

    fig = bqplot.Figure(marks=[scatt, lin, circle, c_point, z_point], axes=[ax_x, ax_y],
                 min_aspect_ratio=1, max_aspect_ratio=1)
    fig.layout.height = '800px'
    return fig

def juliasetNumpy(x, y, c, lim, maxit):
    """ 
    Renvoie l'ensemble de Julia
    
    Paramètres
    ----------
    x: coordonnées des parties réelles regardées
    y: coordonnées des parties imaginaires regardées
    c: nombre complexe figurant dans z^2 + c
    lim: limite du module complexe à partir de laquelle la suite est dite divergente
    maxit: nombre d'itérés maximal
    
    """

    julia = np.zeros((y.size, x.size))
    julia_z = np.zeros((y.size, x.size), dtype='complex')

    zx = x[np.newaxis, :]
    zy = y[:, np.newaxis]
    
    z = zx + 1j*zy
    ite = 0
    while ite < maxit:
        ite += 1
        z_new = z**2 + c
        mask = (np.abs(z) < lim)
        z[mask] = z_new[mask]
        julia[mask] = ite

    return julia, z

def color(z, ite, threshold):
    b1, b2, b3 = 0.4, 0.8, 0.9
    r1, r2, r3 = 0.9, 0.8, 0.4
    
    color = np.empty((*z.shape, 3))
    mask = np.abs(z) < threshold
    color[mask] = [0, 0, 0]
    mask = np.logical_not(mask)
    v = np.zeros_like(ite)
    v[mask] = np.log2(ite[mask] + threshold - np.log2(np.log2(np.abs(z[mask])))) / threshold
    mask = np.logical_and(v < 1.0, np.abs(z) >= threshold)
    color[mask, 0] = v[mask] ** b1
    color[mask, 1] = v[mask] ** b2
    color[mask, 2] = v[mask] ** b3 
    mask = np.logical_and(v >= 1.0, np.abs(z) >= threshold)
    v = np.maximum(0, 2 - v)
    color[mask, 0] = v[mask] ** r1
    color[mask, 1] = v[mask] ** r2
    color[mask, 2] = v[mask] ** r3 
    return color

def julia_plot(c_r=-.92, c_i=-0.22):
    nx = ny = 500

    # limites indiquant quand la suite diverge
    threshold, maxit = 4, 200

    # définition de la grille
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)

    julia, julia_z = juliasetNumpy(x, y, c_r+1j*c_i, threshold, maxit)
    colors = color(julia_z, julia, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(colors)
    ax.axis('off')
    plt.show()