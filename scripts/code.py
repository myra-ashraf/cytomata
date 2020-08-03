

# def process_cad(img_dir, results_dir):
#     """Analyze cad dataset and generate figures."""
#     setup_dirs(os.path.join(results_dir, 'imgs'))
#     t = []
#     ya = []
#     yd = []
#     imgs = []
#     tval = None
#     for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
#         fname = os.path.splitext(os.path.basename(imgf))[0]
#         img, tval = preprocess_img(imgf, tval)
#         cell = segment_object(img, er=7)
#         dots = segment_clusters(img)
#         dots = np.logical_and(dots, cell)
#         anti = np.logical_xor(dots, cell)
#         croi = cell * img
#         aroi = anti * img
#         droi = dots * img
#         cnz = croi[np.nonzero(croi)]
#         anz = aroi[np.nonzero(aroi)]
#         dnz = droi[np.nonzero(droi)]
#         cell_area = len(cnz)/39.0625
#         anti_area = len(anz)/39.0625
#         dots_area = len(dnz)/39.0625
#         if i == 0:
#             cmin = np.min(img)
#             cmax = 1.1*np.max(img)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             anti_int = np.nan_to_num(np.median(anz)) * (anti_area / cell_area)
#             dots_int = np.nan_to_num(np.median(dnz) - np.median(anz)) * (dots_area / cell_area)
#         t.append(np.float(fname))
#         ya.append(anti_int)
#         yd.append(dots_int)
#         with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
#             fig, ax = plt.subplots(figsize=(10,8))
#             axim = ax.imshow(img, cmap='turbo')
#             axim.set_clim(cmin, cmax)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 ax.contour(cell, linewidths=0.4, colors='w')
#                 ax.contour(dots, linewidths=0.2, colors='r')
#             ax.grid(False)
#             ax.axis('off')
#             fig.tight_layout(pad=0)
#             cb = fig.colorbar(axim, pad=0.01, format='%.4f',
#                 extend='both', extendrect=True, extendfrac=0.025)
#             cb.outline.set_linewidth(0)
#             fig.canvas.draw()
#             imgs.append(np.array(fig.canvas.renderer._renderer))
#             fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
#                 dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
#             plt.close(fig)
#     data = np.column_stack((t, ya, yd))
#     np.savetxt(os.path.join(results_dir, 'y.csv'),
#         data, delimiter=',', header='t,ya,yd', comments='')
#     y = np.column_stack((ya, yd))
#     # plot(t, y, xlabel='Time (s)', ylabel='AU',
#     #     labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot.png'))
#     y = np.column_stack((rescale(ya), rescale(yd)))
#     # plot(t, y, xlabel='Time (s)', ylabel='AU',
#     #     labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot01.png'))
#     mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
#     return t, ya, yd


# def combine_cad(results_dir):
#     with plt.style.context(('seaborn-whitegrid', custom_styles)):
#         fig, ax = plt.subplots()
#         combined_ya = pd.DataFrame()
#         combined_yd = pd.DataFrame()
#         for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
#             csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
#             data = pd.read_csv(csv_fp)
#             t = data['t'].values
#             ya = data['ya'].values
#             yd = data['yd'].values
#             yaf = interp1d(t, ya, fill_value='extrapolate')
#             ydf = interp1d(t, yd, fill_value='extrapolate')
#             t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
#             ya = pd.Series([yaf(ti) for ti in t], index=t, name=i)
#             yd = pd.Series([ydf(ti) for ti in t], index=t, name=i)
#             combined_ya = pd.concat([combined_ya, ya], axis=1)
#             combined_yd = pd.concat([combined_yd, yd], axis=1)
#             ax.plot(ya, color='#BBDEFB')
#             ax.plot(yd, color='#ffcdd2')
#         ya_ave = combined_ya.mean(axis=1).rename('ya_ave')
#         ya_std = combined_ya.std(axis=1).rename('ya_std')
#         ya_sem = combined_ya.sem(axis=1).rename('ya_sem')
#         ya_ci = 1.96*ya_sem
#         yd_ave = combined_yd.mean(axis=1).rename('yd_ave')
#         yd_std = combined_yd.std(axis=1).rename('yd_std')
#         yd_sem = combined_yd.sem(axis=1).rename('yd_sem')
#         yd_ci = 1.96*yd_sem
#         combined_data = pd.concat([ya_ave, ya_std, ya_sem, yd_ave, yd_std, yd_sem], axis=1)
#         combined_data.to_csv(os.path.join(results_dir, 'y.csv'))
#         ax.plot(ya_ave, color='#1976D2', label='Anti Region Ave')
#         ax.plot(yd_ave, color='#d32f2f', label='Dots Region Ave')
#         ax.fill_between(t, (ya_ave - ya_ci), (ya_ave + ya_ci), color='#1976D2', alpha=.1)
#         ax.fill_between(t, (yd_ave - yd_ci), (yd_ave + yd_ci), color='#d32f2f', alpha=.1)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('AU')
#         ax.legend(loc='best')
#         fig.savefig(os.path.join(results_dir, 'combined.png'),
#             dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
 

# def linescan(img_dir, results_dir):
#     setup_dirs(os.path.join(results_dir, 'imgs'))
#     setup_dirs(os.path.join(results_dir, 'line'))
#     setup_dirs(os.path.join(results_dir, 'debug'))
#     t = []
#     y = []
#     imgs = []
#     lines = []
#     areas = []
#     for imgf in list_img_files(img_dir):
#         t.append(np.float(os.path.splitext(os.path.basename(imgf))[0]))
#     with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
#         for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
#             fname = os.path.splitext(os.path.basename(imgf))[0]
#             img, raw, bkg = preprocess_img(imgf)
#             if i == 0:
#                 img = rotate(img, angle=-10)
#             if i == 1:
#                 img = rotate(img, angle=-5)
#             if i == 2:
#                 img = rotate(img, angle=-18)
#             if i == 3:
#                 img = rotate(img, angle=-35)
#             if i == 4:
#                 img = rotate(img, angle=-50)
#             if i == 5:
#                 img = rotate(img, angle=-55)
#             thr = segment_object(img, offset=0)
#             roi = thr*img
#             nz = roi[np.nonzero(roi)]
#             ave_int = np.median(nz)
#             y.append(ave_int)
#             labeled = label(thr)
#             rprops = regionprops(labeled)
#             line_row = int(np.round(rprops[0].centroid[0], 0))
#             line_col = int(np.round(rprops[0].centroid[1], 0))
#             area = rprops[0].area
#             areas.append(area)
#             if i == 0:
#                 cmin = np.min(img)
#                 cmax = 1*np.max(img)
#             fig, ax = plt.subplots(figsize=(10,8))
#             axim = ax.imshow(img, cmap='turbo')
#             ax.plot([line_col-40, line_col+40], [line_row, line_row], color='w')
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 ax.contour(thr, linewidths=0.3, colors='w')
#             axim.set_clim(cmin, cmax)
#             ax.grid(False)
#             ax.axis('off')
#             cb = fig.colorbar(axim, pad=0.01, format='%.3f',
#                 extend='both', extendrect=True, extendfrac=0.03)
#             cb.outline.set_linewidth(0)
#             fig.tight_layout(pad=0)
#             fig.canvas.draw()
#             fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
#                 dpi=100, bbox_inches='tight', pad_inches=0)
#             imgs.append(img_as_ubyte(np.array(fig.canvas.renderer._renderer)))
#             plt.close(fig)
#             bg_row = np.argmax(np.var(img, axis=1))
#             fig, ax = plt.subplots(figsize=(10,8))
#             ax.plot(raw[bg_row, :])
#             ax.plot(bkg[bg_row, :])
#             ax.set_title(str(bg_row))
#             fig.savefig(os.path.join(results_dir, 'debug', '{}.png'.format(fname)),
#                 dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#             plt.close(fig)
#             fig, ax = plt.subplots(figsize=(12,8))
#             ax.plot(img[line_row, (line_col-40):(line_col+40)])
#             ax.set_ylabel('Pixel Intensity')
#             ax.set_xlabel('X Position')
#             ax.set_ylim(0, 1)
#             fig.savefig(os.path.join(results_dir, 'line', '{}.png'.format(fname)),
#                 dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#             lines.append(np.array(fig.canvas.renderer._renderer))
#             plt.close(fig)
#         fig, ax = plt.subplots(figsize=(12,8))
#         ax.plot(t, y, color='#d32f2f')
#         ax.set_xlabel('Time (Frame)')
#         ax.set_ylabel('AU')
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig.savefig(os.path.join(results_dir, 'median_intensity.png'),
#             dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
#         fig, ax = plt.subplots(figsize=(12,8))
#         ax.plot(t, areas, color='#d32f2f')
#         ax.set_xlabel('Time (Frame)')
#         ax.set_ylabel('Area (sq. pixels)')
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig.savefig(os.path.join(results_dir, 'area.png'),
#             dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
#     data = np.column_stack((t, y, areas))
#     np.savetxt(os.path.join(results_dir, 'data.csv'),
#         data, delimiter=',', header='t,med_int,area', comments='')
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         mimwrite(os.path.join(results_dir, 'imgs.gif'), imgs, fps=2)
#         mimwrite(os.path.join(results_dir, 'line.gif'), lines, fps=2)
#     return t, y


# def combine_uy(root_dir, fold_change=True):
#     with plt.style.context(('seaborn-whitegrid', custom_styles)):
#         fig, (ax0, ax) = plt.subplots(
#             2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]}
#         )
#         combined_y = pd.DataFrame()
#         combined_uta = pd.DataFrame()
#         combined_utb = pd.DataFrame()
#         for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
#             y_csv_fp = os.path.join(root_dir, data_dir, 'y.csv')
#             y_data = pd.read_csv(y_csv_fp)
#             t = y_data['t'].values
#             y = y_data['y'].values
#             yf = interp1d(t, y, fill_value='extrapolate')
#             t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
#             y = pd.Series([yf(ti) for ti in t], index=t, name=i)
#             if fold_change:
#                 y = y/y[0]
#             combined_y = pd.concat([combined_y, y], axis=1)
#             u_csv_fp = os.path.join(root_dir, data_dir, 'u.csv')
#             u_data = pd.read_csv(u_csv_fp)
#             uta = np.around(u_data['ta'].values, 1)
#             utb = np.around(u_data['tb'].values, 1)
#             uta = pd.Series(uta, name=i)
#             utb = pd.Series(utb, name=i)
#             combined_uta = pd.concat([combined_uta, uta], axis=1)
#             combined_utb = pd.concat([combined_utb, utb], axis=1)
#             ax.plot(y, color='#1976D2', alpha=0.5, linewidth=2)
#         y_ave = combined_y.mean(axis=1).rename('y_ave')
#         y_std = combined_y.std(axis=1).rename('y_std')
#         y_sem = combined_y.sem(axis=1).rename('y_sem')
#         data = pd.concat([y_ave, y_std, y_sem], axis=1).dropna()
#         data.to_csv(os.path.join(root_dir, 'y_combined.csv'))
#         y_ave = data['y_ave']
#         y_ci = data['y_sem']*1.96
#         uta_ave = combined_uta.mean(axis=1).rename('uta_ave')
#         utb_ave = combined_utb.mean(axis=1).rename('utb_ave')
#         tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
#         u = np.zeros_like(tu)
#         for ta, tb in zip(uta, utb):
#             ia = list(tu).index(ta)
#             ib = list(tu).index(tb)
#             u[ia:ib] = 1
#         ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#1976D2', alpha=.2, label='95% CI')
#         ax.plot(y_ave, color='#1976D2', label='Ave')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('AU')
#         if fold_change:
#             ax.set_ylabel('Fold Change')
#         ax.legend(loc='best')
#         ax0.plot(tu, u, color='#1976D2')
#         ax0.set_yticks([0, 1])
#         ax0.set_ylabel('BL')
#         if not fold_change:
#             plot_name = 'y_combined_AU.png'
#         else:
#             plot_name = 'y_combined.png'
#         fig.savefig(os.path.join(root_dir, plot_name),
#             dpi=100, bbox_inches='tight', transparent=False)
#         plt.close(fig)