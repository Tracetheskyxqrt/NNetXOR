using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorBackProp
{
    /// <summary>
    /// Сигмоидальная функция. Используется производная функции для корректировки весов нейронов
    /// </summary>
    abstract class SigmoidFunction
    {
        public static double DerivativeOfSigmoid(double x)
        { //производная сигмоида
            return x * (1 - x);
        }

        public static double output(double x)
        { //сигмоид
            return 1.0 / (1.0 + Math.Exp(-x));
        }       
    }
}
