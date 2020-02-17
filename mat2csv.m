basePath = '/media/emiro593/AdherentCells/';

readPath = strcat(basePath, 'Baxter/Analysis/CellData_200207_123024/');
savePath = strcat(basePath, 'Tracks/');
savePath
mkdir(savePath);

files = dir(fullfile(readPath, '*.mat'));

maxFrames = 0;
for n=1:length(files)
    file = files(n);
    fileName = strcat(readPath, file.name);
    
    T = load(fileName);
    T
    T = T.cellData_compact;
    
    cell = T(1);
    cell
    a = b;
    
     for i=1:length(T)
        cell = T(i);
        
        cell
        
        if (cell.lastFrame > maxFrames)
            maxFrames = cell.lastFrame;
        end
     end
end
    
for n=1:length(files)
    file = files(n);
    fileName = strcat(readPath, file.name);
    
    T = load(fileName);
    T = T.cellData_compact;
        
    X = nan(length(T), maxFrames);
    Y = nan(length(T), maxFrames);
    
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