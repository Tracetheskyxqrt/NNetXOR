using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorBackProp
{
    class Program
    {
        /// <summary>
        /// Точка входа в приложение
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            //сеть включает два входа, два скрытых нейрона и один выходной нерон.
            Console.WriteLine("Веса инициализированы случайными числами. Выберите необходимое количество эпох обучения.");
            Console.WriteLine("Чем больше значение, тем ближе будет результат сети к идеальному и дольше время выполнения. \n");
        StartProgram:
            try
            {
                Console.Write("Введите количество эпох обучения: ");
                int epoch = int.Parse(Console.ReadLine()); //количества эпох обучения
                BackPropagationTrain(epoch); //обучение
            }
            catch (System.FormatException)
            {
                Console.Write("Вы ввели строку в неправильном формате. Пожалуйста, попробуйте снова.");
                goto StartProgram;
            }
            catch (System.OverflowException)
            {
                Console.Write("Вы ввели слишком большое число. Пожалуйста, попробуйте снова. ");
                goto StartProgram;
            }
            catch
            {
                Console.Write("Что-то пошло не так. Пожалуйста, попробуйте снова. ");
                goto StartProgram;
            }
        }

        /// <summary>
        /// Алгоритм обратного распространения ошибки
        /// </summary>
        /// <param name="epoch"></param>
        public static void BackPropagationTrain(int epoch)
        {
            // входные значения. (таблица истинности XOR)
            double[,] inputs =
            {
                {0, 0}, //0 xor 0 = 0
                {0, 1}, //0 xor 1 = 1
                {1, 0}, //1 xor 0 = 1
                {1, 1}  //1 xor 1 = 0
            };

            // желаемые результаты
            double[] DesiredResults = { 0, 1, 1, 0 };

            // Создание нейронов
            Neuron hiddenNeuron1 = new Neuron(); //скрытый нейрон 1
            Neuron hiddenNeuron2 = new Neuron(); //скрытый нейрон 2
            Neuron outputNeuron = new Neuron(); //выход нейрона

            // Инициализация весов случайными числами (от 0.0 до 1.0)
            hiddenNeuron1.InitOfRandomWeights(); //1-ый скрытый
            hiddenNeuron2.InitOfRandomWeights(); //2-оый скрытый нейроны
            outputNeuron.InitOfRandomWeights(); //выход нейрон

            for (int j = 1; j <= epoch; j++) //итерации алгоритма. Цикл от 1 до количества заданных эпох
            {
                for (int i = 0; i < 4; i++) //Важно: не тренироваться только для одного примера до минимизации ошибки
                { //нужно взять каждый пример один раз, а затем начать с начала.
                  // 1) прямое распространение (вычисляет выход)
                    hiddenNeuron1.InputValue = new double[] { inputs[i, 0], inputs[i, 1] };
                    hiddenNeuron2.InputValue = new double[] { inputs[i, 0], inputs[i, 1] };
                    outputNeuron.InputValue = new double[] { hiddenNeuron1.NeuronOutput, hiddenNeuron2.NeuronOutput };
                    Console.WriteLine("{0} xor {1} = {2}  \t|{3} эпоха.", inputs[i, 0], inputs[i, 1], outputNeuron.NeuronOutput, j);
                    // 2) обратное распространение (регулирует веса)
                    // корректирует вес выходного нейрона, основываясь на его ошибке
                    outputNeuron.ErrorValue = SigmoidFunction.DerivativeOfSigmoid(outputNeuron.NeuronOutput) * (DesiredResults[i] - outputNeuron.NeuronOutput);
                    outputNeuron.AdjustmentOfWeights(); //корректировка веса для выходного нейрона
                    // затем корректирует веса скрытых нейронов, основываясь на их ошибках
                    hiddenNeuron1.ErrorValue = SigmoidFunction.DerivativeOfSigmoid(hiddenNeuron1.NeuronOutput) * outputNeuron.ErrorValue * outputNeuron.getInputWeight(0);
                    hiddenNeuron2.ErrorValue = SigmoidFunction.DerivativeOfSigmoid(hiddenNeuron2.NeuronOutput) * outputNeuron.ErrorValue * outputNeuron.getInputWeight(1);
                    hiddenNeuron1.AdjustmentOfWeights(); //корректировка веса для скрытого нейрона 1
                    hiddenNeuron2.AdjustmentOfWeights(); //корректировка веса для скрытого нейрона 2
                }
            }
        }
    }
}
