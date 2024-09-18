import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import streamlit as st

st.subheader('fm. Vaca Muerta')

path = os.path.join(os.getcwd(),'data')
dfprod = pd.read_csv(os.path.join(path,'produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv'), decimal='.')
dffrac = pd.read_csv(os.path.join(path,'datos-de-fractura-de-pozos-de-hidrocarburos-adjunto-iv-actualizacin-diaria.csv'), decimal='.')
dfprod['data'] = dfprod['fecha_data'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
dfprod['qo(m3/d)'] = dfprod['prod_pet']/dfprod['fecha_data'].apply(lambda x:int(x[-2:]))
dfprod['qg(km3/d)'] = dfprod['prod_gas']/dfprod['fecha_data'].apply(lambda x:int(x[-2:]))

# colocando todos os poços na mesma data (m)
datazero = dfprod.groupby('idpozo')['data'].min()
dfprod = dfprod.merge(right=datazero, how='left', on='idpozo', suffixes=('', '_start'))
dfprod['m'] = dfprod['data'].dt.to_period('M').astype('int64') - dfprod['data_start'].dt.to_period('M').astype('int64')

# filtros
dfprod = dfprod[dfprod['formacion']=='vaca muerta']
dfprod = dfprod[dfprod['tipoestado']!='Parado Transitoriamente']

tabFrac, tabHist, tabForecast = st.tabs(['Fraturas', 'Histórico', 'Previsão'])

with tabHist:
    dfprod2 = dfprod
    # listas de bloco e poços
    sAreas = st.multiselect('Bloco', dfprod2['areapermisoconcesion'].unique(), )
    pocos = dfprod2[dfprod2['areapermisoconcesion'].isin(sAreas)]['idpozo'].unique()
    sPocos = st.multiselect(f'Poços({len(pocos)})', pocos)
    if not sPocos: sPocos = pocos
    dfprod2 = dfprod2[(dfprod2['areapermisoconcesion'].isin(sAreas)) & (dfprod2['idpozo'].isin(sPocos))]

    col01, col02, = st.columns(2)
    with col01:
        qoqg = st.selectbox('Vazão: ', options=['qg(km3/d)','qo(m3/d)'])
    with col02:
        leg = st.checkbox('Legenda', value=False)
        m = st.checkbox('Mesma data (ajuste)', value=False)

    if sAreas:
        if m:
            meses = np.arange(0,120,1)
            col11, col12, col13 = st.columns(3)
            with col11:
                decl = st.selectbox('Declínio', options=['Exponencial','Hiperbólico'])
            with col12:
                q0 = st.slider('Q0', 0, 1000, 500, step=20)
            with col13:
                if decl == 'Exponencial':
                    alpha = st.slider('alpha', 0.0, 1.0, 0.05, step=0.1)
                    qAjuste = q0 * np.exp(-meses*alpha)
                else:
                    n = st.slider('n', 0.0, 1.0, 0.5, step=0.05)
                    a = st.slider('a', 0.0, 0.5, 0.1, step=0.01)
                    qAjuste = q0 / np.power(1+n*a*meses,1/n)

            # pot no mes 3
            qAjuste[0] = qAjuste[0]/4
            qAjuste[1] = qAjuste[1]/2
            Np = qAjuste.cumsum()*30.41
            st.write(f'Acum={Np[-1]*1000:,.0f} m³ ({Np[-1]*1000*3.5314666572222e-8:.1f} Bcf)')

            ax = sns.lineplot(data=dfprod2, x='m', hue='idpozo', y=qoqg, palette='tab20')
            plt.plot(meses, qAjuste, lw=2, c='k')
            ax.twinx()
            plt.ylabel('Np(m3) ou Gp(Km3)')
            plt.plot(meses, Np, '--', lw=2, c='k')
            ax.grid()
            if not leg: ax.get_legend().remove()
            st.pyplot(ax.figure, clear_figure=True)

            ax2 = sns.lineplot(data=dfprod2, x='m', y=qoqg, label='média', estimator='mean', errorbar=('pi',50))
            # sns.lineplot(data=dfprod2, x='m', y=qoqg, label='min', estimator='min', ax=ax2)
            # sns.lineplot(data=dfprod2, x='m', y=qoqg, label='max', estimator='max', ax=ax2)
            plt.plot(meses, qAjuste, lw=2, c='k')
            ax.twinx()
            plt.ylabel('Np(m3) ou Gp(Km3)')
            plt.plot(meses, Np, '--', lw=2, c='k')
            ax2.grid()
            st.pyplot(ax2.figure, clear_figure=True)
        else:
            ax = sns.lineplot(data=dfprod2, x='data', hue='idpozo', y=qoqg, palette='tab20')
            ax.grid()
            if not leg: ax.get_legend().remove()
            st.pyplot(ax.figure, clear_figure=True)

with tabFrac:
    c1,c2 = st.columns(2)

    with c1:
        tipo = st.selectbox('Tipo de poço: ', options=['Petrolífero','Gasífero'])
    with c2:
        k = st.selectbox('Tipo de Gráfico: ', options=['scatter', 'kde', 'hist', 'reg'])

    dfprod3 = dfprod[dfprod['tipopozo']==tipo]

    if tipo =='Petrolífero':
        potencial = dfprod3.groupby('idpozo')[['qo(m3/d)','idempresa','areapermisoconcesion',]].max()
        xvars = ['qo(m3/d)']
    else:
        potencial = dfprod3.groupby('idpozo')[['qg(km3/d)','idempresa','areapermisoconcesion',]].max()
        xvars = ['qg(km3/d)']

    potencial = potencial.merge(right=dffrac[['idpozo','longitud_rama_horizontal_m','cantidad_fracturas']], how='left', on='idpozo')
    potencial = potencial[potencial['longitud_rama_horizontal_m']>0]
    xvars.append('longitud_rama_horizontal_m')
    xvars.append('cantidad_fracturas')

    plt.figure()
    empresas = potencial['idempresa'].unique()
    sEmpresa = st.multiselect('Empresa', empresas)
    if not sEmpresa: sEmpresa = empresas
    de = potencial[potencial['idempresa'].isin(sEmpresa)]
    st.pyplot(sns.pairplot(data=de, x_vars=xvars, y_vars=xvars, hue='idempresa', kind=k))

    plt.figure()
    blocos = potencial['areapermisoconcesion'].unique()
    sBloco = st.multiselect('Bloco', blocos)
    if not sBloco: sBloco = blocos
    db = potencial[potencial['areapermisoconcesion'].isin(sBloco)]
    st.pyplot(sns.pairplot(data=db, x_vars=xvars, y_vars=xvars, hue='areapermisoconcesion', kind=k))

with tabForecast:
    col31, col32, col33, col34, col35, col36 = st.columns(6)
    with col31:
        nmax = st.number_input("Total Poços", value=1)
    with col32:
        nSim = st.number_input("x poços", value=1)
    with col33:
        dmeses = st.number_input("a cada (meses)", value=1)
    with col34:
        anos = st.number_input("Horizonte (anos)", value=5)
    with col35:
        qab = st.number_input("Vazão Abandono", value=10, step=10)
    with col36:
        cgr = st.number_input("RGC(m³/MMm³)", value=550)

    mPrev = np.arange(0,anos*12,1)

    if not decl:
        st.write('Faça um ajuste na aba "Histórico"')
    elif decl == 'Exponencial':
        qPrevTipo = q0 * np.exp(-mPrev*alpha)
    else: #'Hiperbólico'
        qPrevTipo = q0 / np.power(1+n*a*mPrev,1/n)
    qPrevTipo = np.asarray([i if i>qab else 0 for i in qPrevTipo])

    qPrev = np.zeros(mPrev.shape[0])

    nWells = 0
    for mes in np.arange(0,mPrev[-1],dmeses):
        if nWells < nmax:
            if mes==0:
                qPrev = nSim*qPrevTipo
            else:
                qPrev[mes:] = nSim*qPrevTipo[:-mes] + qPrev[mes:]
            nWells = nWells + nSim
    NpPrev = qPrev.cumsum()*30.41

    plt.figure()
    plt.grid()
    axPrev = sns.lineplot(x=mPrev,y=qPrev).figure
    plt.ylabel(qoqg)
    plt.twinx()
    plt.ylabel('Acum')
    plt.plot(mPrev,NpPrev)
    st.write(f'Acum={NpPrev[-1]*1000:,.0f} m³ ({NpPrev[-1]*1000*3.5314666572222e-11:.1f} Tcf, {NpPrev[-1]*6.29/1e6:.1f} MMboe)')
    st.pyplot(axPrev, clear_figure=True)

    dfExport = pd.DataFrame({
        'datas': mPrev,
        'Qg': qPrev,
    })
    dfExport
