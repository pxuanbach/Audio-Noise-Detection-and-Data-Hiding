from PyInstaller.utils.hooks import collect_data_files, collect_all
datas, binaries, hiddenimports = collect_all('librosa')
datas += collect_data_files('numba')
hiddenimports += ['numba', 'numba.core', 'numba.np', 'numba.np.ufunc']
binaries += [('C:\\Users\\Admin\\AppData\\Roaming\\Python\\Python312\\site-packages\\numba\\np\\ufunc\\tbb*.dll', 'numba\\np\\ufunc')]