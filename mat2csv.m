basePath = "E:/AdherentIncucyteDemo/Baxter/Analysis/CellData_190319_all_1/Compact/";
savePath = "E:/AdherentIncucyteDemo/Tracks/";

mkdir(savePath);

files = dir(fullfile(basePath, '*.mat'));


for n=1:length(files)
    file = files(n);
    fileName = strcat(basePath, file.name);
    
    T = load(fileName);
    T = T.cellData_compact;
        
    X = nan(length(T), 52);
    Y = nan(length(T), 52);
    
    for i=1:length(T)
        cell = T(i);
        
        ff = cell.firstFrame;
        lf = cell.lastFrame;
        
        cx = cell.cx;
        cy = cell.cy;
        
        X(i, ff:lf) = cx;
        Y(i, ff:lf) = cy;
    end
    
    [filepath, name, ext] = fileparts(fileName);
    
    dlmwrite(strcat(savePath, name, "_X.csv"), X, '\t');
    dlmwrite(strcat(savePath, name, "_Y.csv"), Y, '\t');
end