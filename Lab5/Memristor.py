import numpy as np

class Memristor:
    def __init__(self, D, mu, nu, w0, R_ON, R_OFF, U, Phi):
        self.__D = D
        self.__mu = mu
        self.__nu = nu
        self.__w0 = w0
        self.__R_ON = R_ON
        self.__R_OFF = R_OFF

        self.U = U
        self.Phi = Phi

        self.__w_initial = (
            (self.__w0**2) * (self.__R_ON - self.__R_OFF)
                / (2 * self.__D)
            + self.__w0 * self.__R_OFF
        )

    def w(self, t):
        up_sqrt = (
            self.__R_OFF**2
            + 2 * ((self.__R_ON - self.__R_OFF) / self.__D) * (
                self.__w_initial
                + self.__mu * (self.__R_ON / self.__D) * self.Phi(t))
        )
        up = up_sqrt**.5 - self.__R_OFF
        values = self.__D * up / (self.__R_ON - self.__R_OFF)

        indices = (values >= self.__D) | (values <= 0)
        if not indices.any():
            print('alright')
            return values
        const_index = np.where(indices)[0][0]
        const_value = values[const_index]
        values[const_index:] = self.__D if const_value >= self.__D else 0.0
        return values

    def I(self, t):
        return (self.U(t)
            / (self.__R_ON * self.w(t) / self.__D
               + self.__R_OFF * (1 - self.w(t) / self.__D))
        )

    def q(self, T):
        result = [0]
        last_t = T[0]
        I = self.I(T)
        for t, i in zip(T[1:], I[:-1]):
            result.append(result[-1] + i * (t - last_t))
            last_t = t
        return result[1:]

    def task1(self, plt, N, periods):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        f, axarr = plt.subplots(3, sharex=True)

        for ax in axarr:
            ax.axvline((1 / self.__nu), linewidth=2, color='r', linestyle=':')

        axarr[0].set_title(r'$\frac{\omega\left( t \right)}{D}$')
        axarr[0].plot(TIME, self.w(TIME) / self.__D)
        axarr[0].axhline(1, linewidth=2, color='r', linestyle=':')
        axarr[0].set_ylabel(r'$\frac{\omega}{D}$')

        axarr[1].set_title(r'$I\left( t \right)$')
        axarr[1].plot(TIME, self.I(TIME))
        axarr[1].set_ylabel('$I$')

        axarr[2].set_title(r'$U\left( t \right)$')
        axarr[2].plot(TIME, self.U(TIME))
        axarr[2].set_ylabel('$U$')

        plt.xlabel('$t$')
        plt.show()

    def task2(self, plt, N, periods):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        plt.plot(self.U(TIME), self.I(TIME))
        plt.title(r'$I\left( U \right)$')
        plt.xlabel('$U$')
        plt.ylabel('$I$')
        plt.show()

    def task3(self, plt, N, periods):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        plt.title(r'$q\left( \Phi \right)$')
        plt.xlabel('$\Phi$')
        plt.ylabel('$q$')
        plt.plot(self.Phi(TIME)[1:], self.q(TIME))
        plt.show()
