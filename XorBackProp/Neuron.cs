using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorBackProp
{
    /// <summary>
    /// Класс Нейрона, позволяет оперировать с синаптическими весами
    /// </summary>
    class Neuron
    {
        double[] _inputValue = new double[2]; //входы нейронов
        double[] _synapticWeights = new double[2]; //веса нейронов
        double _errorValue; //значение ошибки
        double _bias; //смещение весов       
        Random _Rand = new Random();

        public double NeuronOutput //выход нейрона
        { //выход: веса предыдущего слоя
            get { return SigmoidFunction.output(this._synapticWeights[0] * this._inputValue[0] + this._synapticWeights[1] * this._inputValue[1] + this._bias); }
        }  

        public void InitOfRandomWeights()
        { //инициализация весов и смещения случайными числами от 0.0 До 1.0
            this._synapticWeights[0] = _Rand.NextDouble();
            this._synapticWeights[1] = _Rand.NextDouble();
            this._bias = _Rand.NextDouble();
        }

        public void AdjustmentOfWeights()
        { //корректировка весов
            this._synapticWeights[0] += this.ErrorValue * this._inputValue[0];
            this._synapticWeights[1] += this.ErrorValue * this._inputValue[1];
            this._bias += this.ErrorValue;
        }

        public double[] InputValue
        {
            set => this._inputValue = value;
        }

        public double ErrorValue
        {
            get
            {
                return this._errorValue;
            }
            set
            {
                this._errorValue = value;
            }
        }

        public double getInputWeight(int index)
        {
            return this._synapticWeights[index];
        }
    }
}
